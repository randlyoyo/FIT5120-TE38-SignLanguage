#!/usr/bin/env python3
"""Extract MediaPipe Holistic keypoints from a folder of sign videos.

Usage:
    python extract.py --videos ./clips --out ./keypoints --model ./holistic_landmarker.task

Writes one .npz per video containing:
    pose        (T, N_POSE_SUB, 3)   normalized image coords, NaN where missing
    pose_vis    (T, N_POSE_SUB)      visibility score
    left_hand   (T, 21, 3)           NaN where the hand was not detected
    right_hand  (T, 21, 3)
    face        (T, N_FACE, 3)
    meta        json blob: fps, size, schema/model versions, detection rates

Design notes that matter:
  * VIDEO running mode, not IMAGE. VIDEO carries tracking state across frames
    and is what the browser will use. IMAGE mode gives different (jumpier)
    numbers and would break train/inference consistency.
  * Missing landmarks are stored as NaN, never as 0.0. Zero is a legal
    coordinate (top-left corner) and the model would learn it as a real pose.
  * NO mirroring is applied here. Frames go in exactly as decoded. See the
    handedness note in the README before you touch this.
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

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _to_array(landmarks, n_expected, subset=None):
    """Convert a MediaPipe landmark list to (n, 3). Returns NaN array if empty."""
    if not landmarks:
        n = len(subset) if subset is not None else n_expected
        return np.full((n, 3), np.nan, dtype=np.float32)
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    if subset is not None:
        if pts.shape[0] <= max(subset):
            n = len(subset)
            return np.full((n, 3), np.nan, dtype=np.float32)
        pts = pts[subset]
    return pts


def _visibility(landmarks, subset):
    if not landmarks:
        return np.full((len(subset),), np.nan, dtype=np.float32)
    vis = np.array(
        [v if (v := getattr(p, "visibility", None)) is not None else np.nan
         for p in landmarks],
        dtype=np.float32,
    )
    if vis.shape[0] <= max(subset):
        return np.full((len(subset),), np.nan, dtype=np.float32)
    return vis[subset]


def extract_video(options, video_path):
    """Extract one clip.

    A fresh landmarker is built per video on purpose. VIDEO running mode keeps
    tracking state and requires strictly increasing timestamps *per instance*;
    reusing one landmarker across clips both leaks clip A's tracking state into
    clip B and makes every clip after the first die with
    "Input timestamp must be monotonically increasing."
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose_l, pose_v, lh_l, rh_l, face_l = [], [], [], [], []
    idx = 0
    prev_ts = -1
    try:
        with HolisticLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=frame_rgb)

                # Timestamps must be strictly increasing integers in ms.
                timestamp_ms = int(round(idx * 1000.0 / fps))
                if timestamp_ms <= prev_ts:
                    timestamp_ms = prev_ts + 1
                prev_ts = timestamp_ms

                res = landmarker.detect_for_video(mp_image, timestamp_ms)

                pose_l.append(
                    _to_array(res.pose_landmarks, C.N_POSE, C.POSE_SUBSET))
                pose_v.append(_visibility(res.pose_landmarks, C.POSE_SUBSET))
                lh_l.append(_to_array(res.left_hand_landmarks, C.N_HAND))
                rh_l.append(_to_array(res.right_hand_landmarks, C.N_HAND))
                face_l.append(
                    _to_array(res.face_landmarks, C.N_FACE_FULL, C.FACE_SUBSET))
                idx += 1
    finally:
        cap.release()

    if idx == 0:
        raise RuntimeError(f"decoded 0 frames from {video_path}")

    pose = np.stack(pose_l)
    left_hand = np.stack(lh_l)
    right_hand = np.stack(rh_l)

    def rate(arr):
        return float(np.mean(~np.isnan(arr[:, 0, 0])))

    meta = {
        "schema_version": C.SCHEMA_VERSION,
        "mediapipe_version": mp.__version__,
        "model_tag": C.MODEL_TAG,
        "running_mode": "VIDEO",
        "source": video_path.name,
        "fps": float(fps),
        "width": width,
        "height": height,
        "frames_decoded": idx,
        "frames_declared": declared_frames,
        "pose_rate": rate(pose),
        "left_hand_rate": rate(left_hand),
        "right_hand_rate": rate(right_hand),
        "mirrored": False,
    }

    return {
        "pose": pose,
        "pose_vis": np.stack(pose_v),
        "left_hand": left_hand,
        "right_hand": right_hand,
        "face": np.stack(face_l),
        "meta": meta,
    }


_OPTS = None


def _init_worker(model_path):
    """One options object per worker process. Built here rather than pickled --
    MediaPipe objects do not survive a fork cleanly."""
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


def _save(d, dst):
    np.savez_compressed(
        dst,
        pose=d["pose"], pose_vis=d["pose_vis"],
        left_hand=d["left_hand"], right_hand=d["right_hand"],
        face=d["face"], meta=json.dumps(d["meta"]),
    )


def _job_path(task):
    """Worker for --videos mode: (source video, destination npz)."""
    src, dst = task
    try:
        d = extract_video(_OPTS, Path(src))
        _save(d, Path(dst))
        return (dst, d["meta"], None)
    except Exception as e:                    # noqa: BLE001
        return (dst, None, str(e))


def _job_zip(task, zip_path=None):
    """Worker for --zip mode. Unpacks ONE clip to a temp file, extracts, deletes.

    Peak disk stays at the zip plus a single clip, so a 14 GB archive never has
    to be unpacked. Each worker opens its own ZipFile handle -- a shared handle
    is not safe across processes.
    """
    entry, dst = task
    tmp = None
    try:
        with zipfile.ZipFile(zip_path) as zf, \
                tempfile.NamedTemporaryFile(suffix=Path(entry).suffix,
                                            delete=False) as fh:
            tmp = fh.name
            fh.write(zf.read(entry))
        d = extract_video(_OPTS, Path(tmp))
        d["meta"]["source"] = entry
        _save(d, Path(dst))
        return (dst, d["meta"], None)
    except Exception as e:                    # noqa: BLE001
        return (dst, None, str(e))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos", type=Path, help="directory of video files")
    src.add_argument("--zip", type=Path, help="read videos straight from a .zip")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path,
                    help="path to holistic_landmarker.task")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help="process at most N clips")
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f"model bundle not found: {args.model}")
    args.out.mkdir(parents=True, exist_ok=True)

    # Build the task list. Output is flat: one .npz per clip, named after the
    # clip, so the clip id stays usable as a join key for labels later.
    if args.zip:
        with zipfile.ZipFile(args.zip) as zf:
            entries = sorted(n for n in zf.namelist()
                             if Path(n).suffix.lower() in VIDEO_EXTS)
        if not entries:
            sys.exit(f"no videos inside {args.zip}")
        pairs = [(e, args.out / (Path(e).stem + ".npz")) for e in entries]
        worker = partial(_job_zip, zip_path=str(args.zip))
        label = args.zip.name
    else:
        vids = sorted(p for p in args.videos.rglob("*")
                      if p.suffix.lower() in VIDEO_EXTS)
        if not vids:
            sys.exit(f"no videos found under {args.videos}")
        pairs = [(str(v), args.out / (v.stem + ".npz")) for v in vids]
        worker = _job_path
        label = str(args.videos)

    total = len(pairs)
    if not args.overwrite:
        pairs = [(a, b) for a, b in pairs if not b.exists()]
    if args.limit:
        pairs = pairs[: args.limit]
    tasks = [(a, str(b)) for a, b in pairs]

    print(f"{label}: {total} clips, {total - len(tasks)} already done, "
          f"{len(tasks)} to do, {args.workers} worker(s)")
    if not tasks:
        return

    failures, done = [], 0
    with Pool(args.workers, initializer=_init_worker,
              initargs=(str(args.model),)) as pool:
        for dst, meta, err in pool.imap_unordered(worker, tasks, chunksize=4):
            done += 1
            name = Path(dst).name
            if err:
                failures.append((name, err))
                print(f"[{done}/{len(tasks)}] FAIL {name}: {err}", flush=True)
            elif done % 25 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] {name}  "
                      f"frames={meta['frames_decoded']:4d} "
                      f"pose={meta['pose_rate']:.2f} "
                      f"L={meta['left_hand_rate']:.2f} "
                      f"R={meta['right_hand_rate']:.2f}", flush=True)

    print(f"\ndone: {done - len(failures)} ok, {len(failures)} failed")
    for name, err in failures[:20]:
        print(f"  {name}: {err}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
