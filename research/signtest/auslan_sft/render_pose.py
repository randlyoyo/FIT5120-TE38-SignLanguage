#!/usr/bin/env python3
"""Render a pose sequence as a stick-figure MP4.

    python render_pose.py --pose outputs/example.npy --output outputs/example.mp4

Accepted inputs:
  .npy  (T, 79, 2) signer-space pose; NaN marks a missing joint. A sidecar
        <name>.json (written by infer.py) supplies fps and view bounds.
  .pt   a preprocessed sample from normalize_pose.py ({"pose", "mask", "fps"}),
        i.e. ground truth -- useful to check the renderer against real data.

The camera is fixed for the whole video: from the checkpoint's/sidecar's
dataset view bounds when available, otherwise from the sequence's own 0.5-99.5
percentile extent, computed once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slp_models import skeleton as sk  # noqa: E402
from slp_utils.visualization import camera_for, render_video  # noqa: E402


def load_pose(path: Path):
    """Returns (pose (T,J,2), valid (T,J) or None, meta dict)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    meta: dict = {}
    if path.suffix == ".npy":
        pose = np.load(path)
        side = path.with_suffix(".json")
        if side.is_file():
            meta = json.loads(side.read_text())
        valid = None
    elif path.suffix == ".pt":
        import torch
        s = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(s, dict) and "pose" in s:
            pose = s["pose"].numpy()
            valid = s["mask"].numpy() if "mask" in s else None
            meta = {"fps": s.get("fps", 25.0), "text": s.get("text")}
        else:
            pose, valid = np.asarray(s), None
    else:
        raise ValueError(f"unsupported pose file {path.suffix}; use .npy or .pt")
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 3 or pose.shape[1:] != (sk.NUM_JOINTS, sk.COORD_DIM):
        raise ValueError(f"expected pose (T, {sk.NUM_JOINTS}, {sk.COORD_DIM}), got {pose.shape}")
    if pose.shape[0] == 0:
        raise ValueError("empty pose sequence")
    return pose, valid, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="stick-figure MP4 renderer")
    ap.add_argument("--pose", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=float, default=None, help="default: sidecar/.pt fps, else 25")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--compare", default=None, help="second pose file rendered side by side (e.g. ground truth)")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--auto-camera", action="store_true", help="ignore stored view bounds")
    args = ap.parse_args(argv)

    pose, valid, meta = load_pose(Path(args.pose))
    panels = [(pose, valid, meta.get("label", "generated" if args.compare else None))]
    if args.compare:
        p2, v2, m2 = load_pose(Path(args.compare))
        panels.append((p2, v2, "reference"))
    fps = args.fps or float(meta.get("fps", 25.0))
    bounds = None if args.auto_camera else meta.get("view_bounds")
    cam = camera_for([p for p, _, _ in panels], bounds, args.size)
    out = render_video(Path(args.output), panels, fps, cam, args.caption or meta.get("text"))
    print(f"wrote {out} ({max(p.shape[0] for p, _, _ in panels)} frames @ {fps:g} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
