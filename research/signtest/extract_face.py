#!/usr/bin/env python3
"""Blendshape coefficients for the takes the dictionary actually uses.

    python extract_face.py --workers 4

extract.py kept 18 of MediaPipe's 478 face points -- twelve on the outer lip
and six on the brows -- which is enough to tell that a brow moved and not much
else. An avatar wants blendshape weights, and FaceLandmarker produces the 52
ARKit-compatible ones directly, including the eye and jaw channels the 18-point
subset cannot express at all.

Only the 3,215 chosen takes are processed, not all 83,590 clips: the dictionary
plays one recording per word, so the rest would be work for nothing.

The frame is cropped to the head before detection. On the full 512x408 frame
FaceLandmarker finds nothing at all -- the signer's face spans about fifty
pixels -- and simply upscaling the whole frame does not help either: the
detector wants the face to fill a good part of the image, not merely to be
large in pixels. Cropping around the ears and nose, which the pose landmarks
already give, and resizing to 256 takes detection from 0/9 frames to 9/9.
Blendshape weights are expression ratios and so survive the crop unchanged,
and the crop is an axis-aligned translation and scale, which leaves the head
rotation in the transformation matrix intact.

Two things carried over from the earlier extraction, both learned the hard way:
mediapipe is pinned at 0.10.33 because 1.0.x aborts the process on Apple
silicon in a way try/except cannot catch, and a landmarker instance is built
per clip because VIDEO mode requires monotonically rising timestamps -- sharing
one across clips silently yields results for the first video only, with an exit
code of zero.
"""
import argparse
import json
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import slr_common as C

P11 = {v: i for i, v in enumerate(C.POSE_SUBSET)}
OUT = Path("renders/face")
SRC = Path("renders/src")
MODEL = "face_landmarker.task"
ZIPS = {"Train": "MM-WLAuslan/Train/rgb.zip", "Valid": "MM-WLAuslan/Valid/rgb.zip",
        "Test_STU": "MM-WLAuslan/Test_STU/rgb.zip",
        "Test_ITW": "MM-WLAuslan/Test_ITW/rgb.zip",
        "Test_TED": "MM-WLAuslan/Test_TED/rgb.zip"}


def ensure_mp4(split, stem):
    p = SRC / f"{stem}.mp4"
    if not p.exists():
        SRC.mkdir(parents=True, exist_ok=True)
        z = zipfile.ZipFile(ZIPS[split])
        name = next(n for n in z.namelist() if stem in n)
        with z.open(name) as f, open(p, "wb") as o:
            o.write(f.read())
    return p


def head_crop(frame, pose2, t, size=256, pad=2.2):
    """Square crop around the head, from the ears and nose already detected."""
    if t >= pose2.shape[0]:
        return None
    import cv2
    h, w = frame.shape[:2]
    pts = pose2[t, [P11[7], P11[8], P11[0]]]
    if not np.isfinite(pts).all():
        return None
    ear = pts[:2] * [w, h]
    r = float(np.linalg.norm(ear[0] - ear[1])) * pad
    if not np.isfinite(r) or r < 8:
        return None
    cx, cy = (pts.mean(axis=0) * [w, h])
    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
    x1, y1 = int(min(w, cx + r)), int(min(h, cy + r))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return cv2.resize(frame[y0:y1, x0:x1], (size, size))


def one(job):
    word, split, stem = job
    out = OUT / f"{word.replace('/', '_')}.json"
    if out.exists():
        return dict(word=word, skipped=True)
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        path = ensure_mp4(split, stem)
        kp = np.load(f"keypoints/{split}/{stem}.npz", allow_pickle=False)
        pose2 = kp["pose"][:, :, :2]
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        names, rows, mats = None, [], []
        with vision.FaceLandmarker.create_from_options(opts) as lm:
            t = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                crop = head_crop(frame, pose2, t)
                if crop is None:
                    rows.append(None); mats.append(None); t += 1
                    continue
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                r = lm.detect_for_video(img, int(t * 1000.0 / fps))
                t += 1
                if r.face_blendshapes:
                    bs = r.face_blendshapes[0]
                    if names is None:
                        names = [c.category_name for c in bs]
                    rows.append([round(c.score, 4) for c in bs])
                else:
                    rows.append(None)
                if r.facial_transformation_matrixes:
                    mats.append(np.array(r.facial_transformation_matrixes[0]))
                else:
                    mats.append(None)
        cap.release()
        if names is None:
            return dict(word=word, error="no face detected in any frame")

        # Hold the previous frame through a dropout rather than writing zeros:
        # a zero blendshape vector is a NEUTRAL face, so a gap would render as
        # the signer's expression snapping off and back.
        filled, last, missing = [], [0.0] * len(names), 0
        for r in rows:
            if r is None:
                missing += 1
                filled.append(list(last))
            else:
                last = r
                filled.append(r)

        # Head pose from the transformation matrix, which is a real rotation
        # rather than the ear-and-nose construction face_channels.py needs.
        head = []
        lastm = [0.0, 0.0, 0.0]
        for m in mats:
            if m is None:
                head.append(list(lastm))
                continue
            R = m[:3, :3]
            sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
            if sy > 1e-6:
                ang = [np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy),
                       np.arctan2(R[1, 0], R[0, 0])]
            else:
                ang = [np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0]
            lastm = [round(float(np.degrees(a)), 2) for a in ang]
            head.append(list(lastm))

        OUT.mkdir(parents=True, exist_ok=True)
        json.dump({"word": word, "source": f"{split}/{stem}", "fps": fps,
                   "names": names, "blendshapes": filled,
                   "headRPY": head, "missing": missing,
                   "frames": len(filled)},
                  open(out, "w"), separators=(",", ":"))
        return dict(word=word, frames=len(filled), missing=missing)
    except Exception as e:
        return dict(word=word, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    man = json.load(open("renders/manifest.json"))["words"]
    if a.words:
        want = set(a.words)
        man = [w for w in man if w["word"] in want]
    if a.limit:
        man = man[:a.limit]
    jobs = [(w["word"], w["split"], w["stem"]) for w in man]
    print(f"{len(jobs)} takes -> {OUT}", flush=True)
    ok = skip = 0
    bad = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            if "error" in r:
                bad.append(r)
            elif r.get("skipped"):
                skip += 1
            else:
                ok += 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(jobs)}  ok {ok}  failed {len(bad)}", flush=True)
    print(f"\n{ok} written, {skip} skipped, {len(bad)} failed")
    for r in bad[:6]:
        print("  ", r["word"], r["error"])


if __name__ == "__main__":
    main()
