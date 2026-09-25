"""English text -> Auslan signing, with the SignSparK models fine-tuned in auslan_smplx/.

Same pipeline as signspark_render.generate() (EXPERIMENTS.md E26-E34, the colab_auslan_generate
notebook), restructured for a server: generate() builds the three stream models for every call and
re-fits the retrieval index each time, which costs minutes. Here everything is built once at startup:

  * the three stream models (hand / body / face) stay loaded; each one's frozen text encoder is
    moved to the CPU after priming, as signspark_ft.build does, so the GPU holds only the generators;
  * the retrieval bank (the training LMDB), its word-count -> length fit, its TF-IDF index and its
    M-CLIP embeddings are computed once.

Per sentence the steps and numbers are unchanged: normalise + <Auslan> tag; length from the word
count; nearest training sentence by TF-IDF (+1e-3 x M-CLIP cosine to break ties); hands sampled
from that sentence's rescaled keyframes; body and face from the text alone unless the sentence is
in the training set word for word; Gaussian smoothing sigma 1 per stream.

Output per sentence: the SMPL-X parameters of every frame (axis-angle, what an avatar rig needs),
the 127 SMPL-X joint positions, and optionally a skeleton mp4.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import torch

STREAMS = ("hand", "body", "face")
POSE_KEYS = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose",
             "leye_pose", "reye_pose", "expression", "betas", "transl")


class SignGenerator:
    def __init__(self, cfg: dict, device: torch.device, log=print):
        c = cfg["text2sign"]
        self.cfg = c
        self.device = device
        self.amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.lock = threading.Lock()
        t0 = time.time()

        # SignSparK stays on sys.path for the life of the process (its sampler may import lazily);
        # Uni-Sign is the one loaded in a scope, see _imports.py.
        os.environ.setdefault("WANDB_MODE", "disabled")
        if c["code_dir"] not in sys.path:
            sys.path.insert(0, c["code_dir"])
        import signspark_ft as ft
        ft.setup_paths(c["signspark_repo"])
        import signspark_render as R
        self.ft, self.R = ft, R

        self.bank = R.load_bank(c["bank_lmdb"])
        self.bank_texts = [e["text"] for e in self.bank]
        self.bank_set = set(self.bank_texts)
        self.len_a, self.len_b = R.fit_length([R._untag(t) for t in self.bank_texts], [e["T"] for e in self.bank])
        from sklearn.feature_extraction.text import TfidfVectorizer          # = R.word_sim, fitted once
        self.tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, sublinear_tf=True)
        self.bank_tfidf = self.tfidf.fit_transform([R._untag(t) for t in self.bank_texts])
        log(f"[text2sign] retrieval bank: {len(self.bank)} clips, length = {self.len_a:.1f} + {self.len_b:.2f} x words")

        self.models = {}
        for s in STREAMS:
            cfg_s = ft.load_cfg(c["signspark_repo"], s)
            prime = sorted(self.bank_set) if s == "hand" else ["<Auslan> hello ."]
            self.models[s] = ft.build(cfg_s, os.path.join(c["weights_dir"], f"{s}.pt"), device, texts=prime)
        enc = self.models["hand"][0].encode_text
        self.bank_emb = torch.nn.functional.normalize(enc(self.bank_texts).float(), dim=-1)
        self.skeleton = R.Skeleton(c["smplx_npz"], c["signspark_repo"], device)
        log(f"[text2sign] SignSparK hand/body/face ready on {device}, bf16 {self.amp} ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ generation
    def _retrieve(self, text: str) -> int:
        R = self.R
        q = self.tfidf.transform([R._untag(text)])
        score = (q @ self.bank_tfidf.T).toarray().astype(np.float32)[0]
        emb = torch.nn.functional.normalize(self.models["hand"][0].encode_text([text]).float(), dim=-1)
        score = score + 1e-3 * (emb @ self.bank_emb.T).cpu().numpy()[0]
        return int(score.argmax())

    def features(self, sentence: str) -> dict:
        """One sentence -> hand (T, 180), body (T, 60), face (T, 56) in the loader's convention."""
        R, c = self.R, self.cfg
        text = f"<Auslan> {R.normalise(sentence)}"
        T = R.est_length(R._untag(text), self.len_a, self.len_b)
        seen = text in self.bank_set
        entry = self.bank[self._retrieve(text)]
        out = {}
        for s in STREAMS:
            model, flow = self.models[s]
            if s == "hand" or seen:
                arr, _ = R.sample_keyframed(model, flow, [R.keyframe_batch(s, [text], [T], [entry])],
                                            self.amp, self.device, c["steps"], c["seed"])
            else:
                arr = R.sample_text_only(model, flow, [R.text_batch(s, [text], [T])],
                                         self.amp, self.device, c["steps"], c["text_scale"], c["seed"])
            out[s] = R.smooth_stream(s, R.per_clip(s, arr, [T])[0], c["sigma"])
        return {"sentence": sentence, "text": text, "retrieved": R._untag(entry["text"]), "seen": seen, "T": T, **out}

    @torch.no_grad()
    def smplx(self, f: dict) -> tuple[dict, np.ndarray]:
        """Features -> (SMPL-X forward inputs as numpy, joints (T, 127, 3)). Same call as Skeleton.joints."""
        sk = self.skeleton
        kw = sk.build(torch.from_numpy(f["body"]).float(), torch.from_numpy(f["hand"][:, :90]).float(),
                      torch.from_numpy(f["hand"][:, 90:]).float(), face=torch.from_numpy(f["face"]).float(),
                      device=sk.device)
        joints = sk.model(**kw).joints.cpu().numpy()
        params = {k: v.detach().float().cpu().numpy() for k, v in kw.items() if torch.is_tensor(v)}
        return params, joints

    def generate(self, sentence: str, out_dir: str, name: str) -> dict:
        """Sentence -> files in out_dir: <name>.json (pose for the avatar) and <name>.mp4 (optional)."""
        import json
        t0 = time.time()
        with self.lock:                     # one generation on the GPU at a time
            f = self.features(sentence)
            params, joints = self.smplx(f)
        t1 = time.time()
        pose = {
            "format": "smplx-v1",
            "fps": self.R.FPS,
            "frames": int(f["T"]),
            "sentence": sentence,
            "retrieved": f["retrieved"],
            "seen": bool(f["seen"]),
            "rotation": "axis-angle (radians), SMPL-X joint order; flat_hand_mean=True, 50 FLAME-2020 expression coefficients",
            "smplx": {k: np.round(v, 5).tolist() for k, v in params.items() if k in POSE_KEYS},
            "joints": np.round(joints, 4).tolist(),
            "parents": self.skeleton.parents.tolist(),
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
            json.dump(pose, fh, separators=(",", ":"))
        video = None
        if self.cfg["render_video"]:
            video = f"{name}.mp4"
            self.R.write_video(os.path.join(out_dir, video), [joints], ["Auslan"], self.skeleton.parents,
                               size=self.cfg["video_size"], caption=sentence)
        return {"pose_file": f"{name}.json", "video_file": video, "frames": int(f["T"]), "fps": self.R.FPS,
                "retrieved": f["retrieved"], "seen": bool(f["seen"]),
                "timings": {"generate": round(t1 - t0, 3), "render": round(time.time() - t1, 3)}}
