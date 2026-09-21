#!/usr/bin/env python3
"""Extract metric 3D keypoints (world landmarks) alongside the 2D ones.

    python extract3d.py --zip MM-WLAuslan/Train/rgb.zip --out keypoints3d/Train \
                        --model ./holistic_landmarker.task --workers 4

This is a SEPARATE extractor. extract.py and the keypoints/ tree it produced are
untouched: the 2D pipeline, its cached features and the trained encoder all keep
working exactly as they are.

Why redo extraction at all
--------------------------
The 2D baseline collapses on Test_MTV (AUC 0.59, rank-1 0.9%) and the diagnosis
was viewpoint, not detection quality: shoulder foreshortening runs 0.73-3.10
there against a tight 1.68-1.82 on every frontal split. A 2D projection of an
off-axis signer is a different signal, and no amount of translating or scaling
in the image plane recovers it.

World landmarks are in metres, so a clip can be rotated about the vertical axis
to a canonical facing before anything is compared. That is the one principled
fix available for a viewpoint change.

What is stored
--------------
    pose_world        (T, 33, 3)  metres, origin at the hip midpoint
    left_hand_world   (T, 21, 3)  metres, in the hand's OWN local frame
    right_hand_world  (T, 21, 3)
    pose              (T, 11, 3)  normalised image coords, POSE_SUBSET
    pose_vis          (T, 11)
    left_hand         (T, 21, 3)  normalised image coords
    right_hand        (T, 21, 3)
    face              (T, 18, 3)  normalised image coords, FACE_SUBSET
    meta              json

The normalised arrays are byte-identical in meaning to what extract.py writes,
so a 3D .npz is a strict superset of a 2D one and nothing has to be extracted a
third time.

The full 33-point pose is kept rather than POSE_SUBSET's 11: hips are the origin
of the world frame and the torso plane is what makes a rotation well defined, so
throwing them away here would defeat the purpose. Subsetting stays a decision
for the feature builder.

NO canonicalisation happens here. Raw world coordinates go to disk, exactly as
extract.py stores raw image coordinates -- rotation alignment belongs in the
dataloader where it can change without a 26-hour re-run.

Note: MediaPipe Holistic exposes no world landmarks for the face, so the face
stays 2D-only. That is a limitation of the model, not a choice.
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    RunningMode,
)

import slr_common as C

# Separate from slr_common.SCHEMA_VERSION so the 2D contract is not disturbed.
SCHEMA_VERSION_3D = "3.0.0"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
N_POSE_FULL = C.N_POSE          # 33


def _xyz(landmarks, n, subset=None):
    """Landmark list -> (n,3) float32, NaN when the group was not detected."""
    if not landmarks:
        return np.full((len(subset) if subset is not None else n, 3),
                       np.nan, dtype=np.float32)
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    if subset is not None:
        if pts.shape[0] <= max(subset):
            return np.full((len(subset), 3), np.nan, dtype=np.float32)
        pts = pts[subset]
    elif pts.shape[0] != n:
        return np.full((n, 3), np.nan, dtype=np.float32)
    return pts


def _visibility(landmarks, subset):
    if not landmarks:
        return np.full((len(subset),), np.nan, dtype=np.float32)
    vis = np.array(
        [v if (v := getattr(p, "visibility", None)) is not None else np.nan
         for p in landmarks], dtype=np.float32)
    if vis.shape[0] <= max(subset):
        return np.full((len(subset),), np.nan, dtype=np.float32)
    return vis[subset]


def extract_video(options, video_path):
    """One clip -> dict of arrays.

    A fresh landmarker per video, for the same reason extract.py does it: VIDEO
    mode keeps tracking state and demands strictly increasing timestamps per
    instance, so reusing one across clips both leaks state and kills every clip
    after the first.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    acc = {k: [] for k in ("pw", "lhw", "rhw", "p", "pv", "lh", "rh", "f")}
    idx, prev_ts = 0, -1
    try:
        with HolisticLandmarker.create_from_options(options) as lm:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ts = int(round(idx * 1000.0 / fps))
                if ts <= prev_ts:
                    ts = prev_ts + 1
                prev_ts = ts
                r = lm.detect_for_video(img, ts)

                # --- metric 3D ---
                acc["pw"].append(_xyz(r.pose_world_landmarks, N_POSE_FULL))
                acc["lhw"].append(_xyz(r.left_hand_world_landmarks, C.N_HAND))
                acc["rhw"].append(_xyz(r.right_hand_world_landmarks, C.N_HAND))
                # --- normalised 2D/2.5D, same contract as extract.py ---
                acc["p"].append(_xyz(r.pose_landmarks, N_POSE_FULL, C.POSE_SUBSET))
                acc["pv"].append(_visibility(r.pose_landmarks, C.POSE_SUBSET))
                acc["lh"].append(_xyz(r.left_hand_landmarks, C.N_HAND))
                acc["rh"].append(_xyz(r.right_hand_landmarks, C.N_HAND))
                acc["f"].append(_xyz(r.face_landmarks, C.N_FACE_FULL,
                                     C.FACE_SUBSET))
                idx += 1
    finally:
        cap.release()

    if idx == 0:
        raise RuntimeError(f"decoded 0 frames from {video_path}")

    out = {k: np.stack(v) for k, v in acc.items()}

    def rate(a):
        return float(np.mean(~np.isnan(a[:, 0, 0])))

    meta = {
        "schema_version": SCHEMA_VERSION_3D,
        "schema_version_2d": C.SCHEMA_VERSION,
        "mediapipe_version": mp.__version__,
        "model_tag": C.MODEL_TAG,
        "running_mode": "VIDEO",
        "source": video_path.name,
        "fps": float(fps), "width": width, "height": height,
        "frames_decoded": idx, "frames_declared": declared,
        "pose_rate": rate(out["p"]),
        "pose_world_rate": rate(out["pw"]),
        "left_hand_rate": rate(out["lh"]),
        "right_hand_rate": rate(out["rh"]),
        "left_hand_world_rate": rate(out["lhw"]),
        "right_hand_world_rate": rate(out["rhw"]),
        "mirrored": False,
        "canonicalised": False,   # rotation alignment is the dataloader's job
    }
    return out, meta


def _save(out, meta, dst):
    np.savez_compressed(
        dst,
        pose_world=out["pw"],
        left_hand_world=out["lhw"],
        right_hand_world=out["rhw"],
        pose=out["p"], pose_vis=out["pv"],
        left_hand=out["lh"], right_hand=out["rh"], face=out["f"],
        meta=json.dumps(meta),
    )


_OPTS = None


def _init(model_path):
    global _OPTS
    _OPTS = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        min_face_detection_confidence=0.5,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )


def _job_zip(task, zip_path=None):
    """Unpack ONE clip to a temp file, extract, delete. Peak disk stays at the
    archive plus a single video, so a 14 GB zip never has to be expanded."""
    entry, dst = task
    tmp = None
    try:
        with zipfile.ZipFile(zip_path) as zf, \
                tempfile.NamedTemporaryFile(suffix=Path(entry).suffix,
                                            delete=False) as fh:
            tmp = fh.name
            fh.write(zf.read(entry))
        out, meta = extract_video(_OPTS, Path(tmp))
        meta["source"] = entry
        _save(out, meta, Path(dst))
        return (dst, meta, None)
    except Exception as e:                                  # noqa: BLE001
        return (dst, None, str(e))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def _job_path(task):
    src, dst = task
    try:
        out, meta = extract_video(_OPTS, Path(src))
        _save(out, meta, Path(dst))
        return (dst, meta, None)
    except Exception as e:                                  # noqa: BLE001
        return (dst, None, str(e))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos", type=Path)
    src.add_argument("--zip", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f"model bundle not found: {args.model}")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.zip:
        with zipfile.ZipFile(args.zip) as zf:
            entries = sorted(n for n in zf.namelist()
                             if Path(n).suffix.lower() in VIDEO_EXTS)
        if not entries:
            sys.exit(f"no videos inside {args.zip}")
        pairs = [(e, args.out / (Path(e).stem + ".npz")) for e in entries]
        worker, label = partial(_job_zip, zip_path=str(args.zip)), args.zip.name
    else:
        vids = sorted(p for p in args.videos.rglob("*")
                      if p.suffix.lower() in VIDEO_EXTS)
        if not vids:
            sys.exit(f"no videos found under {args.videos}")
        pairs = [(str(v), args.out / (v.stem + ".npz")) for v in vids]
        worker, label = _job_path, str(args.videos)

    total = len(pairs)
    if not args.overwrite:
        pairs = [(a, b) for a, b in pairs if not b.exists()]
    if args.limit:
        pairs = pairs[: args.limit]
    tasks = [(a, str(b)) for a, b in pairs]

    print(f"{label}: {total} clips, {total - len(tasks)} done, "
          f"{len(tasks)} to do, {args.workers} worker(s)")
    if not tasks:
        return

    failures, done = [], 0
    with Pool(args.workers, initializer=_init, initargs=(str(args.model),)) as pool:
        for dst, meta, err in pool.imap_unordered(worker, tasks, chunksize=4):
            done += 1
            name = Path(dst).name
            if err:
                failures.append((name, err))
                print(f"[{done}/{len(tasks)}] FAIL {name}: {err}", flush=True)
            elif done % 25 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] {name}  "
                      f"frames={meta['frames_decoded']:4d} "
                      f"pose3d={meta['pose_world_rate']:.2f} "
                      f"L3d={meta['left_hand_world_rate']:.2f} "
                      f"R3d={meta['right_hand_world_rate']:.2f}", flush=True)

    print(f"\ndone: {done - len(failures)} ok, {len(failures)} failed")
    for n, e in failures[:20]:
        print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
