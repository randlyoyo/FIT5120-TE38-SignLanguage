#!/usr/bin/env python3
"""Video -> raw whole-body keypoints (.npz), one file per annotation row.

Estimator: rtmlib `Wholebody(mode="lightweight")` (RTMW-l-m + YOLOX-tiny), 133
COCO-WholeBody keypoints with 21 points per hand. This is deliberately NOT
MediaPipe Holistic: it is the estimator the pretrained Uni-Sign checkpoint was
trained on, so the frozen pretrained pose encoder (feature loss) and the SLR
back-translation evaluator see in-distribution input. Switching estimators
would silently invalidate both.

Output format is identical to unisign/extract_pose.py, so existing Auslan-Daily
extractions (Drive: auslan_work/pose/chunk_*.tar) can be dropped into
data.raw_pose_dir under their uid and this step skipped entirely.

    keypoints (T, 133, 2) float32, pixel coordinates divided by [W, H]
    scores    (T, 133)    float32
    meta      JSON: width, height, fps, n_frames, estimator, person_select

Person selection: the largest detection box per frame (measured 99.97 % frame
accuracy on Auslan-Daily; see unisign/extract_pose.py for the study).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.make_splits import read_annotations, sample_id  # noqa: E402
from slp_utils.config import load_config, resolve_path  # noqa: E402


def read_video(path: Path, max_frames: int | None = None):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded from {path}")
    h, w = frames[0].shape[:2]
    return frames, w, h, fps


class WholebodyExtractor:
    def __init__(self, device: str = "cpu", backend: str = "onnxruntime", mode: str = "lightweight"):
        try:
            from rtmlib import Wholebody
        except ImportError as exc:
            raise ImportError("rtmlib is required: pip install rtmlib onnxruntime(-gpu)") from exc
        self.model = Wholebody(mode=mode, backend=backend, device=device, to_openpose=False)
        self.tag = f"rtmlib.Wholebody/{mode}"

    def __call__(self, frames, width: int, height: int):
        T = len(frames)
        kp_out = np.zeros((T, 133, 2), dtype=np.float32)
        sc_out = np.zeros((T, 133), dtype=np.float32)
        wh = np.asarray([width, height], dtype=np.float32)
        for t, frame in enumerate(frames):
            boxes = self.model.det_model(frame)
            if len(boxes) == 0:
                # rtmlib's own fallback (kept for parity with the Uni-Sign
                # extraction): run the pose model on the whole image.
                boxes = np.asarray([[0, 0, width, height]], dtype=np.float32)
            kps, scores = self.model.pose_model(frame, bboxes=boxes)
            kps, scores = np.asarray(kps, dtype=np.float32), np.asarray(scores, dtype=np.float32)
            if kps.ndim == 2:
                kps, scores = kps[None], scores[None]
            b = np.asarray(boxes, dtype=np.float64)
            i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
            if kps.shape[1] != 133:
                raise RuntimeError(f"estimator returned {kps.shape[1]} keypoints, expected 133")
            kp_out[t] = kps[i] / wh
            sc_out[t] = scores[i]
        return kp_out, sc_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="extract whole-body keypoints from videos")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th row starting at i")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    rows = read_annotations(resolve_path(cfg, d["annotations_csv"]), d)
    video_root = resolve_path(cfg, d["video_root"])
    out_dir = resolve_path(cfg, d["raw_pose_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    si, sn = (int(x) for x in args.shard.split("/"))
    rows = rows[si::sn]
    if args.limit:
        rows = rows[: args.limit]

    ex = WholebodyExtractor(device=args.device)
    done = failed = skipped = 0
    t0 = time.time()
    for row in rows:
        sid = sample_id(row, d)
        dst = out_dir / f"{sid}.npz"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        vp = Path(row[d.get("video_column", "video_path")])
        vp = vp if vp.is_absolute() else video_root / vp
        try:
            frames, w, h, fps = read_video(vp)
            kp, sc = ex(frames, w, h)
        except Exception as exc:
            failed += 1
            print(f"[extract] FAILED {sid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        meta = {"uid": sid, "source": str(vp), "width": w, "height": h, "fps": fps,
                "n_frames": len(frames), "estimator": ex.tag, "person_select": "largest",
                "coordinate_space": "frame_normalised"}
        tmp = dst.with_name(dst.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, keypoints=kp, scores=sc, meta=json.dumps(meta))
        tmp.replace(dst)
        done += 1
        print(f"[extract] {sid}: {len(frames)} frames {w}x{h}@{fps:.1f} "
              f"({done} done, {time.time()-t0:.0f}s)", flush=True)
    print(f"[extract] done={done} skipped={skipped} failed={failed}")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
