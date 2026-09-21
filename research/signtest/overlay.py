#!/usr/bin/env python3
"""Draw extracted keypoints back onto the source video so you can eyeball them.

Usage:
    python overlay.py --video ./clips/HOUSE_01.mp4 --npz ./keypoints/HOUSE_01.npz \
                      --out ./checks/HOUSE_01_overlay.mp4

Or batch:
    python overlay.py --videos ./clips --npz-dir ./keypoints --out-dir ./checks --limit 20

Colour code (memorise this, it is the point of the whole script):
    LEFT hand  = BLUE
    RIGHT hand = ORANGE
Raise only your right hand in a test clip. If the orange skeleton does not
appear on the hand you actually raised, you have a mirroring bug.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import slr_common as C

COL_POSE = (200, 200, 200)
COL_LEFT = (255, 120, 0)     # BGR -> blue
COL_RIGHT = (0, 140, 255)    # BGR -> orange
COL_FACE = (140, 220, 140)
COL_BAD = (0, 0, 255)


def _px(pt, w, h):
    if np.isnan(pt[0]) or np.isnan(pt[1]):
        return None
    return (int(round(pt[0] * w)), int(round(pt[1] * h)))


def _draw_set(img, pts, connections, colour, w, h, radius=3):
    coords = [_px(p, w, h) for p in pts]
    for a, b in connections:
        if coords[a] and coords[b]:
            cv2.line(img, coords[a], coords[b], colour, 2, cv2.LINE_AA)
    for c in coords:
        if c:
            cv2.circle(img, c, radius, colour, -1, cv2.LINE_AA)
    return sum(1 for c in coords if c)


def render(video_path, npz_path, out_path):
    d = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    pose, lh, rh, face = d["pose"], d["left_hand"], d["right_hand"], d["face"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or meta.get("fps", 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    i = 0
    n = pose.shape[0]
    while True:
        ok, frame = cap.read()
        if not ok or i >= n:
            break

        _draw_set(frame, pose[i], C.POSE_CONNECTIONS, COL_POSE, w, h, radius=4)
        n_l = _draw_set(frame, lh[i], C.HAND_CONNECTIONS, COL_LEFT, w, h)
        n_r = _draw_set(frame, rh[i], C.HAND_CONNECTIONS, COL_RIGHT, w, h)
        # Lips as a closed ring (mouth shape is a non-manual marker you want to
        # be able to read); brows as plain dots.
        _draw_set(frame, face[i][:C.N_LIPS], C.LIP_CONNECTIONS,
                  COL_FACE, w, h, radius=1)
        for p in face[i][C.N_LIPS:]:
            c = _px(p, w, h)
            if c:
                cv2.circle(frame, c, 2, COL_FACE, -1, cv2.LINE_AA)

        hud = [
            f"frame {i+1}/{n}",
            f"L(blue): {'ok' if n_l else 'MISSING'}",
            f"R(orange): {'ok' if n_r else 'MISSING'}",
        ]
        for k, line in enumerate(hud):
            colour = COL_BAD if "MISSING" in line else (255, 255, 255)
            cv2.putText(frame, line, (10, 24 + k * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, 24 + k * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)

        writer.write(frame)
        i += 1

    cap.release()
    writer.release()

    if i != n:
        print(f"  WARNING: video had {i} usable frames but npz has {n}. "
              f"Frame counts must match -- investigate before trusting anything.")
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--videos", type=Path)
    ap.add_argument("--npz-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    if args.video:
        frames = render(args.video, args.npz, args.out)
        print(f"wrote {args.out} ({frames} frames)")
        return

    if not (args.videos and args.npz_dir and args.out_dir):
        ap.error("give either --video/--npz/--out or --videos/--npz-dir/--out-dir")

    npzs = sorted(args.npz_dir.rglob("*.npz"))[: args.limit]
    for p in npzs:
        rel = p.relative_to(args.npz_dir)
        candidates = list(args.videos.rglob(rel.stem + ".*"))
        candidates = [c for c in candidates if c.suffix.lower() != ".npz"]
        if not candidates:
            print(f"no source video for {rel}")
            continue
        dst = args.out_dir / rel.with_name(rel.stem + "_overlay.mp4")
        frames = render(candidates[0], p, dst)
        print(f"wrote {dst} ({frames} frames)")


if __name__ == "__main__":
    main()
