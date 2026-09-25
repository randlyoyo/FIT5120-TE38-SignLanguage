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

import json
import os
import sys
import threading
import time

import numpy as np
import torch

STREAMS = ("hand", "body", "face")
POSE_KEYS = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose",
             "leye_pose", "reye_pose", "expression", "betas", "transl")


class _slim_weights_allowed:
    """While building, let signspark_ft.load_weights accept a slim file (scripts/slim_assets.py):
    keys listed in <stream>.dropped.json are missing from it because a fresh build already holds
    exactly those values. Without the sidecar the original strict check runs unchanged."""

    def __init__(self, ft, sidecar):
        self.ft, self.sidecar = ft, sidecar

    def __enter__(self):
        if not os.path.exists(self.sidecar):
            return
        ft, dropped = self.ft, set(json.load(open(self.sidecar)))
        self.orig = ft.load_weights

        def load(model, path):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            bad = [k for k in missing if not k.startswith(ft.TEXT_KEY) and k not in dropped]
            assert not bad and not unexpected, f"{path}: missing {bad[:5]}, unexpected {unexpected[:5]}"
        ft.load_weights = load

    def __exit__(self, *exc):
        if hasattr(self, "orig"):
            self.ft.load_weights = self.orig


# --------------------------------------------------------------------------- compact retrieval bank
def compact_bank(bank: list[dict]) -> dict:
    """signspark_render.load_bank entries -> arrays holding each clip's keyframe rows only (what
    keyframe_batch reads). Frame 0 is kept for clips under 2 frames, which warped_keyframes maps to it."""
    names, texts, T, offsets, frames = [], [], [], [0], []
    rows = {s: [] for s in STREAMS}
    for e in bank:
        kf = sorted(set(e["keyframes"]) | ({0} if e["T"] < 2 else set()))
        names.append(e["name"]); texts.append(e["text"]); T.append(e["T"])
        frames += kf
        offsets.append(offsets[-1] + len(kf))
        for s in STREAMS:
            rows[s].append(e[s][kf])
    return {"names": np.array(names), "texts": np.array(texts), "T": np.array(T, np.int32),
            "offsets": np.array(offsets, np.int64), "frames": np.array(frames, np.int32),
            "keyframes_json": np.array(json.dumps([e["keyframes"] for e in bank])),
            **{s: np.concatenate(rows[s]).astype(np.float32) for s in STREAMS}}


def load_compact_bank(path: str) -> list[dict]:
    z = np.load(path)
    kfs = json.loads(str(z["keyframes_json"]))
    out = []
    for i in range(len(z["texts"])):
        a, b = int(z["offsets"][i]), int(z["offsets"][i + 1])
        out.append({"name": str(z["names"][i]), "text": str(z["texts"][i]), "T": int(z["T"][i]), "keyframes": kfs[i],
                    "kf_frames": z["frames"][a:b], **{s: z[s][a:b] for s in STREAMS}, "compact": True})
    return out


def expand_entry(e: dict) -> dict:
    """A compact entry -> full-length stream arrays that are right at every keyframe (the only rows
    keyframe_batch reads) and zero elsewhere. Full LMDB entries pass through unchanged."""
    if not e.get("compact"):
        return e
    full = dict(e)
    for s in STREAMS:
        arr = np.zeros((max(e["T"], 1), e[s].shape[1]), np.float32)
        arr[e["kf_frames"]] = e[s]
        full[s] = arr
    return full


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

        compact = c["bank_lmdb"].endswith(".npz")         # scripts/slim_assets.py output, or the LMDB itself
        self.bank = load_compact_bank(c["bank_lmdb"]) if compact else R.load_bank(c["bank_lmdb"])
        self.bank_texts = [e["text"] for e in self.bank]
        self.bank_set = set(self.bank_texts)
        self.len_a, self.len_b = R.fit_length([R._untag(t) for t in self.bank_texts], [e["T"] for e in self.bank])
        from sklearn.feature_extraction.text import TfidfVectorizer          # = R.word_sim, fitted once
        self.tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, sublinear_tf=True)
        self.bank_tfidf = self.tfidf.fit_transform([R._untag(t) for t in self.bank_texts])
        log(f"[text2sign] retrieval bank: {len(self.bank)} clips, length = {self.len_a:.1f} + {self.len_b:.2f} x words")

        stored_emb = None
        if compact:
            with np.load(c["bank_lmdb"]) as z:
                stored_emb = z["emb"] if "emb" in z.files else None
        self.models = {}
        for s in STREAMS:
            cfg_s = ft.load_cfg(c["signspark_repo"], s)
            prime = sorted(self.bank_set) if s == "hand" and stored_emb is None else ["<Auslan> hello ."]
            with _slim_weights_allowed(ft, os.path.join(c["weights_dir"], f"{s}.dropped.json")):
                self.models[s] = ft.build(cfg_s, os.path.join(c["weights_dir"], f"{s}.pt"), device, texts=prime)
        if c.get("share_text_encoder", True):
            self._share_text_encoder()
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[c.get("weights_dtype", "float32")]
        if dtype != torch.float32:
            self._cast_generators(dtype)
        if c.get("encoder_on_gpu") and device.type == "cuda":
            # build() parks the frozen 2.2 GB text encoder on the CPU; with GPU memory to spare each new
            # sentence is then encoded on the GPU instead (tens of ms instead of ~0.5 s)
            for s in STREAMS:
                self.models[s][0].text_enc_model.to(device)
        if stored_emb is not None:                       # computed by slim_assets.py exactly as below
            self.bank_emb = torch.from_numpy(stored_emb).to(self.models["hand"][0].embed_text.weight.device)
        else:
            enc = self.models["hand"][0].encode_text
            self.bank_emb = torch.nn.functional.normalize(enc(self.bank_texts).float(), dim=-1)
        self.skeleton = R.Skeleton(c["smplx_npz"], c["signspark_repo"], device)
        log(f"[text2sign] SignSparK hand/body/face ready on {device}, bf16 autocast {self.amp}, "
            f"weights {getattr(self, 'weights_dtype', 'float32')}, shared text encoder "
            f"{getattr(self, 'shared_text_encoder', False)} ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ memory
    def _share_text_encoder(self):
        """One text encoder for all three streams instead of three copies (2.2 GB each).

        Each stream model loads the same frozen M-CLIP encoder from Hugging Face when it is built and
        never trains it (our weight files exclude it), so the three copies should be identical. That is
        checked tensor by tensor before sharing; sharing identical weights cannot change any output."""
        ref = self.models["hand"][0].text_enc_model
        ref_sd = ref.state_dict()
        for s in ("body", "face"):
            model = self.models[s][0]
            sd = model.text_enc_model.state_dict()
            bad = [k for k in ref_sd if k not in sd or not torch.equal(ref_sd[k].cpu(), sd[k].cpu())]
            if bad or len(sd) != len(ref_sd):
                raise RuntimeError(f"text encoders differ between hand and {s} ({bad[:3]}); not sharing them")
            model.text_enc_model = ref
        self.shared_text_encoder = True
        import gc
        gc.collect()

    def _cast_generators(self, dtype):
        """Store each stream's generator (the UNet and its small embeddings) in `dtype`, halving its
        memory. The text encoder stays fp32, so sentence embeddings are unchanged. Sampling already runs
        under bf16 autocast, so the arithmetic is the same kind; what changes is that the weights
        themselves are rounded to bf16 (scripts/check_bf16.py measures the effect)."""
        if not self.amp:
            raise RuntimeError(f"weights_dtype {dtype} needs bf16 autocast (a CUDA GPU with bf16)")
        # Real floating tensors only. Module.to(dtype) would also cast complex buffers to the real dtype,
        # dropping their imaginary part (the first bf16 run did that to a complex buffer in the UNet:
        # "Casting complex values to real discards the imaginary part").
        cast = lambda t: t.to(dtype) if t.is_floating_point() else t
        for s in STREAMS:
            model = self.models[s][0]
            for name, child in model.named_children():
                if name != "text_enc_model":
                    child._apply(cast)
            for name, p in list(model._parameters.items()):
                if p is not None and p.is_floating_point():
                    p.data = p.data.to(dtype)
            for name, b in list(model._buffers.items()):
                if b is not None and b.is_floating_point():
                    model._buffers[name] = b.to(dtype)
        self.weights_dtype = str(dtype).replace("torch.", "")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ generation
    def _retrieve(self, text: str) -> int:
        R = self.R
        q = self.tfidf.transform([R._untag(text)])
        score = (q @ self.bank_tfidf.T).toarray().astype(np.float32)[0]
        emb = torch.nn.functional.normalize(self.models["hand"][0].encode_text([text]).float(), dim=-1)
        score = score + 1e-3 * (emb @ self.bank_emb.T).cpu().numpy()[0]
        return int(score.argmax())

    def features(self, sentence: str, parallel: bool | None = None) -> dict:
        """One sentence -> hand (T, 180), body (T, 60), face (T, 56) in the loader's convention.

        parallel (config `parallel_streams`): the three streams are independent models, so they are
        sampled at the same time, each in its own thread and CUDA stream, instead of one after the
        other. The result is meant to be identical: each stream's noise is drawn up front exactly as
        signspark_render.sample_* draws it (fork_rng + manual_seed(seed * 1000) + randn_like), because
        the global RNG must not be seeded from three threads at once; the ODE solve itself is
        deterministic. parallel=False runs signspark_render's own sample_* functions, the reference
        that scripts/check_parallel.py compares against."""
        R, c = self.R, self.cfg
        parallel = c.get("parallel_streams", True) if parallel is None else parallel
        text = f"<Auslan> {R.normalise(sentence)}"
        T = R.est_length(R._untag(text), self.len_a, self.len_b)
        seen = text in self.bank_set
        entry = expand_entry(self.bank[self._retrieve(text)])
        keyed = {s: s == "hand" or seen for s in STREAMS}
        batches = {s: R.keyframe_batch(s, [text], [T], [entry]) if keyed[s] else R.text_batch(s, [text], [T])
                   for s in STREAMS}
        if parallel and self.device.type == "cuda":
            arrays = self._sample_parallel(batches, keyed)
        else:
            arrays = {}
            for s in STREAMS:
                model, flow = self.models[s]
                if keyed[s]:
                    arrays[s], _ = R.sample_keyframed(model, flow, [batches[s]], self.amp, self.device, c["steps"], c["seed"])
                else:
                    arrays[s] = R.sample_text_only(model, flow, [batches[s]], self.amp, self.device,
                                                   c["steps"], c["text_scale"], c["seed"])
        out = {s: R.smooth_stream(s, R.per_clip(s, arrays[s], [T])[0], c["sigma"]) for s in STREAMS}
        return {"sentence": sentence, "text": text, "retrieved": R._untag(entry["text"]), "seen": seen, "T": T, **out}

    def _sample_parallel(self, batches: dict, keyed: dict) -> dict:
        """signspark_render.sample_keyframed / sample_text_only for one batch per stream, run concurrently.
        Same inputs, noise, masks, guidance and solver; only the noise is drawn before the threads start."""
        from concurrent.futures import ThreadPoolExecutor
        R, ft, c, dev = self.R, self.ft, self.cfg, self.device
        jobs = {}
        for s in STREAMS:                                   # sequential: device transfer, masks, noise
            x, y = batches[s]
            self.models[s][0].encode_text(sorted(set(y["text"])))   # cache it: one shared encoder, no thread races
            x = x.to(dev)
            y = dict(y, mask=y["mask"].to(dev))
            if keyed[s]:
                B, J, T = x.shape
                obs = ft.kf_mask(y["keyframes"], B, J, T).to(dev).contiguous()
            else:
                obs = torch.zeros(x.shape, dtype=torch.bool, device=dev)
            with torch.random.fork_rng(devices=[dev]):
                torch.manual_seed(c["seed"] * 1000)         # batch index 0, as in sample_*
                noise = torch.randn_like(x)
            jobs[s] = (x, y, obs, noise)
        if not hasattr(self, "_streams"):
            self._streams = {s: torch.cuda.Stream(device=dev) for s in STREAMS}
            self._pool = ThreadPoolExecutor(len(STREAMS), thread_name_prefix="signspark")
        main = torch.cuda.current_stream(dev)

        def run(s):
            model, flow = self.models[s]
            x, y, obs, noise = jobs[s]
            cs = self._streams[s]
            cs.wait_stream(main)                            # inputs were made on the main stream
            with torch.no_grad(), torch.cuda.stream(cs):    # grad mode and the stream are per thread
                ft.set_mode(model, False)
                fwd = ft.Amp(model, self.amp)
                if keyed[s]:
                    fn = fwd
                else:
                    def fn(xx, t, y=None, obs_x0=None, obs_mask=None):
                        o = fwd(xx, t, y=y, obs_x0=obs_x0, obs_mask=obs_mask)
                        u = fwd(xx, t, y=dict(y, uncond=True), obs_x0=obs_x0, obs_mask=obs_mask)
                        return u + c["text_scale"] * (o - u)
                out, _ = flow.decode(fn, noise=noise, keyframe_mask=obs, x_embed=x,
                                     model_kwargs={"y": y, "obs_x0": x, "obs_mask": obs},
                                     ode_package="torchdiffeq", ode_stepnum=c["steps"])
                return [out.float().cpu().numpy()]         # .cpu() waits for this stream

        return dict(zip(STREAMS, self._pool.map(run, STREAMS)))

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
        """Sentence -> <name>.json in out_dir (pose for the avatar), and <name>.mp4 (skeleton video).

        With render_video "async" (the default) the reply returns as soon as the pose JSON is written
        and the mp4 is drawn by a background thread (CPU and ffmpeg only, ~0.6-1.6 s); it appears at
        video_url when done (video_status "rendering"; 404 until then). True renders before replying,
        False never."""
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
        tmp = os.path.join(out_dir, f".{name}.json.part")
        with open(tmp, "w") as fh:
            json.dump(pose, fh, separators=(",", ":"))
        os.replace(tmp, os.path.join(out_dir, f"{name}.json"))

        mode = self.cfg["render_video"]
        video, status = (f"{name}.mp4", "ready") if mode else (None, "off")
        if mode == "async":
            if not hasattr(self, "_render_pool"):
                from concurrent.futures import ThreadPoolExecutor
                self._render_pool = ThreadPoolExecutor(2, thread_name_prefix="render")
            self._render_pool.submit(self._render, out_dir, name, joints, sentence)
            status = "rendering"
        elif mode:
            self._render(out_dir, name, joints, sentence)
        return {"pose_file": f"{name}.json", "video_file": video, "video_status": status, "frames": int(f["T"]),
                "fps": self.R.FPS, "retrieved": f["retrieved"], "seen": bool(f["seen"]),
                "timings": {"generate": round(t1 - t0, 3), "write": round(time.time() - t1, 3)}}

    def _render(self, out_dir, name, joints, sentence):
        """Draw to a hidden temporary name and rename, so video_url never serves a half-written file."""
        tmp = os.path.join(out_dir, f".{name}.part.mp4")
        try:
            self.R.write_video(tmp, [joints], ["Auslan"], self.skeleton.parents,
                               size=self.cfg["video_size"], caption=sentence)
            os.replace(tmp, os.path.join(out_dir, f"{name}.mp4"))
        except Exception as e:              # a failed video must not take the server down
            print(f"[text2sign] rendering {name}.mp4 failed: {type(e).__name__}: {e}", flush=True)
            for leftover in (tmp, tmp + ".tmp.mp4"):        # write_video's own intermediate file
                if os.path.exists(leftover):
                    os.remove(leftover)
