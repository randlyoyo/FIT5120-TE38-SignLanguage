"""Auslan video -> English text, with the Uni-Sign model trained in unisign/.

The input path is exactly the one the model was trained and scored on:
  rtmlib Wholebody 'lightweight', largest person per frame, keypoints / [W, H]   (extract_pose.py)
  -> spec.load_part_kp (part selection, confidence 0.3, normalisation)          (spec.py)
  -> uniform stride to at most 256 frames (the deterministic eval subsample)    (dataset.py)
  -> Uni_Sign.forward + mT5 beam search, num_beams 4                            (train.py UniSignBackend)

Two differences from the recorded clips a live video can have, handled here:
  * frame rate: Auslan-Daily is 25 fps; a 30/60 fps webcam clip is resampled to 25 so the signing
    speed matches what the model saw;
  * mirroring: front cameras often save a mirror image, which turns a right-handed signer into a
    left-handed one (mirrored clips were excluded from training). The client says so with
    `mirrored=True` and the frames are flipped back.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ._imports import scoped_path

SHOULDERS = (5, 6)          # COCO-WholeBody indices, used to tell whether a signer is in view


class SignRecognizer:
    def __init__(self, cfg: dict, device: torch.device, log=print):
        c = cfg["sign2text"]
        self.cfg = c
        self.device = device
        t0 = time.time()
        with scoped_path(c["code_dir"], c["unisign_repo"]):
            import dataset
            import extract_pose
            import model_adapter
            import spec
            self.spec, self.dataset, self.extract_pose = spec, dataset, extract_pose
            if c["pose_device"] == "cuda":
                try:        # onnxruntime-gpu >= 1.21: load the CUDA / cuDNN libraries torch's wheels ship
                    import onnxruntime
                    onnxruntime.preload_dlls()
                except Exception:
                    pass
            self.extractor = extract_pose.PoseExtractor(device=c["pose_device"], pose_batch=c["pose_batch"])
            if c["pose_device"] == "cuda":
                self.set_cudnn_algo(c.get("pose_cudnn_algo", "HEURISTIC"))
            self.model = model_adapter.load_unisign(c["checkpoint"], c["unisign_repo"], c["mt5_path"],
                                                    device=str(device)).eval()
            # One pass through the whole model while the repo is still importable, so any import it
            # does on first use happens now (see _imports.py).
            self._decode(self._batch(np.zeros((8, 133, 2), np.float32), np.ones((8, 133), np.float32)))
        # onnxruntime silently falls back to the CPU when CUDA cannot be loaded; say what actually runs
        wb = self.extractor._wholebody
        self.pose_providers = sorted({p for m in (wb.det_model, wb.pose_model)
                                      for p in (getattr(getattr(m, "session", None), "get_providers", lambda: [])()[:1])})
        log(f"[sign2text] Uni-Sign ready on {device} ({time.time() - t0:.0f}s), "
            f"pose model {self.extractor.model_tag} requested {c['pose_device']}, running on {self.pose_providers}"
            f"{', cudnn ' + self.cudnn_algo if getattr(self, 'cudnn_algo', None) else ''}")

    def set_cudnn_algo(self, algo: str) -> None:
        """Rebuild the detector and pose sessions with onnxruntime's cudnn_conv_algo_search = algo.
        onnxruntime's default, EXHAUSTIVE, benchmarks every convolution algorithm for each new input
        shape, and in steady state the algorithms it picked for these depthwise models were slow:
        27 ms/frame for the pose model on an A100 against 3.8 ms with HEURISTIC; 40 vs 16 ms/frame
        overall (scripts/profile_pose.py). Confident keypoints moved by 0.02 px median (p99 2.5 px),
        the chosen person never changed; scripts/check_pose_setting.py compares the translations."""
        import onnxruntime as ort
        opts = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": algo}), "CPUExecutionProvider"]
        wb = self.extractor._wholebody
        for m in (wb.det_model, wb.pose_model):
            m.session = ort.InferenceSession(m.session._model_path, providers=opts)
        self.cudnn_algo = algo

    # ------------------------------------------------------------------ steps
    def read_video(self, path: str, mirrored: bool = False):
        import cv2
        c = self.cfg
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("cannot open the uploaded video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not 1.0 <= fps <= 240.0:          # webm from browsers often reports 0 or 1000
            fps = float(c["target_fps"])
        frames = []
        cap_frames = int(c["max_seconds"] * fps) + 1
        while len(frames) < cap_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            raise ValueError("the uploaded video has no decodable frames")
        if fps > c["target_fps"] + 0.5:
            keep = np.round(np.arange(0, len(frames), fps / c["target_fps"])).astype(int)
            frames = [frames[i] for i in keep[keep < len(frames)]]
        if mirrored:
            frames = [cv2.flip(f, 1) for f in frames]
        h, w = frames[0].shape[:2]
        return frames, w, h, fps

    def _batch(self, kp: np.ndarray, sc: np.ndarray) -> dict:
        keep = self.dataset._subsample(len(kp), self.cfg["max_length"], deterministic=True)
        parts = self.spec.load_part_kp(kp[keep], sc[keep])
        item = {"uid": "live", "dataset": "live", "subset": "live", "text": "", "length": len(keep)}
        item.update({p: torch.from_numpy(parts[p]) for p in self.spec.PART_ORDER})
        return self.dataset.collate([item])

    @torch.no_grad()
    def _decode(self, batch: dict) -> str:
        src = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()
               if k in self.spec.PART_ORDER or k in ("attention_mask", "src_length_batch", "name_batch")}
        tgt = {"gt_sentence": [""], "gt_gloss": [""]}
        out = self.model(src, tgt)
        ids = self.model.mt5_model.generate(inputs_embeds=out["inputs_embeds"], attention_mask=out["attention_mask"],
                                            max_new_tokens=self.cfg["max_new_tokens"], num_beams=self.cfg["num_beams"])
        return self.model.mt5_tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()

    # ------------------------------------------------------------------ public
    def recognise(self, video_path: str, mirrored: bool = False) -> dict:
        t0 = time.time()
        frames, w, h, fps = self.read_video(video_path, mirrored)
        t1 = time.time()
        kp, sc, audit = self.extractor.run(frames, w, h)
        t2 = time.time()
        visible = float(np.mean((audit["n_people"] > 0) & (sc[:, list(SHOULDERS)].min(1) > self.spec.CONF_THRESHOLD)))
        if visible < self.cfg["min_person_frames"]:
            raise ValueError(f"no signer found: upper body visible in {visible:.0%} of frames "
                             f"(need {self.cfg['min_person_frames']:.0%}); face the camera with both shoulders in view")
        text = self._decode(self._batch(kp, sc))
        return {"text": text, "frames": len(frames), "source_fps": round(fps, 2), "size": [w, h],
                "signer_visible": round(visible, 3),
                "timings": {"decode_video": round(t1 - t0, 3), "pose": round(t2 - t1, 3),
                            "translate": round(time.time() - t2, 3)}}
