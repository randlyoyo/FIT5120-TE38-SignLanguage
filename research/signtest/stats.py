#!/usr/bin/env python3
"""Numeric sanity checks over a directory of extracted .npz keypoint files.

Usage:
    python stats.py --npz-dir ./keypoints --csv ./checks/quality.csv

Reports, per clip and in aggregate:
  * frame count, and whether it matches what the decoder declared
  * pose / left-hand / right-hand detection rate
  * dominant-hand rate (the busier hand -- the one that actually matters)
  * coordinate range violations (normalized coords should sit near [0,1])
  * frozen-frame count (identical consecutive frames = decoder or tracker stall)
  * a flag list, so you can pull the bad clips and watch them in overlay.py

This tells you the extraction did not crash. It does NOT tell you the
keypoints are useful -- for that you need the class-separation test.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import slr_common as C


def rate(arr):
    """Fraction of frames where this landmark group was detected at all."""
    return float(np.mean(~np.isnan(arr[:, 0, 0])))


def longest_gap(arr):
    """Longest run of consecutive frames with the group missing."""
    missing = np.isnan(arr[:, 0, 0])
    best = cur = 0
    for m in missing:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return int(best)


def motion_energy(arr):
    """Mean per-frame displacement, ignoring missing frames. Used to decide
    which hand is dominant and to catch clips where nothing moves."""
    valid = ~np.isnan(arr[:, 0, 0])
    if valid.sum() < 2:
        return 0.0
    pts = arr[valid][:, :, :2]
    d = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    return float(np.nanmean(d))


def analyse(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    pose, lh, rh = d["pose"], d["left_hand"], d["right_hand"]
    T = pose.shape[0]

    l_rate, r_rate = rate(lh), rate(rh)
    l_energy, r_energy = motion_energy(lh), motion_energy(rh)

    # Dominant hand = the one that carries the sign: seen often AND moving.
    # Score both together -- a tuple compare is lexicographic, so rate alone
    # would decide and a few spurious frames on the idle hand could flip it.
    if l_rate * (l_energy + 1e-6) >= r_rate * (r_energy + 1e-6):
        dom_rate, dom_side = l_rate, "left"
    else:
        dom_rate, dom_side = r_rate, "right"

    # Normalized coords can drift slightly outside [0,1] when a limb leaves
    # frame; far outside means something is wrong.
    xy = np.concatenate([pose[:, :, :2], lh[:, :, :2], rh[:, :, :2]], axis=1)
    with np.errstate(invalid="ignore"):
        oob = float(np.nanmean((xy < -0.2) | (xy > 1.2)))

    # Frozen frames: pose identical to the previous frame.
    pose_flat = np.nan_to_num(pose.reshape(T, -1), nan=-999.0)
    frozen = int(np.sum(np.all(np.diff(pose_flat, axis=0) == 0, axis=1)))

    flags = []
    if T < C.MIN_FRAMES:
        flags.append("too_short")
    if meta.get("frames_declared", T) not in (T, 0, -1) and \
            abs(meta["frames_declared"] - T) > 2:
        flags.append("frame_count_mismatch")
    if rate(pose) < C.MIN_POSE_RATE:
        flags.append("low_pose")
    if dom_rate < C.MIN_DOMINANT_HAND_RATE:
        flags.append("low_dominant_hand")
    if longest_gap(lh) > T * 0.4 and longest_gap(rh) > T * 0.4:
        flags.append("long_both_hand_gap")
    if oob > 0.02:
        flags.append("out_of_bounds")
    if frozen > T * 0.1:
        flags.append("frozen_frames")
    if max(l_energy, r_energy) < 1e-4:
        flags.append("no_motion")
    if np.isnan(pose).all():
        flags.append("all_nan")

    return {
        "clip": path.stem,
        "frames": T,
        "frames_declared": meta.get("frames_declared", -1),
        "fps": round(meta.get("fps", 0.0), 2),
        "pose_rate": round(rate(pose), 3),
        "left_rate": round(l_rate, 3),
        "right_rate": round(r_rate, 3),
        "dominant": dom_side,
        "dominant_rate": round(dom_rate, 3),
        "left_gap": longest_gap(lh),
        "right_gap": longest_gap(rh),
        "oob_frac": round(oob, 4),
        "frozen": frozen,
        "flags": "|".join(flags),
        "schema": meta.get("schema_version", "?"),
        "mediapipe": meta.get("mediapipe_version", "?"),
    }


def histogram(values, bins=10, width=40):
    if not values:
        return
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    top = max(counts) or 1
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(c / top * width))
        print(f"  {lo:.1f}-{hi:.1f} | {bar} {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True, type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    paths = sorted(args.npz_dir.rglob("*.npz"))
    if not paths:
        raise SystemExit(f"no .npz under {args.npz_dir}")

    rows = [analyse(p) for p in paths]

    schemas = {r["schema"] for r in rows}
    mps = {r["mediapipe"] for r in rows}

    print(f"\n{len(rows)} clips\n")
    print("Dominant-hand detection rate distribution:")
    histogram([r["dominant_rate"] for r in rows])

    print("\nPose detection rate distribution:")
    histogram([r["pose_rate"] for r in rows])

    flagged = [r for r in rows if r["flags"]]
    print(f"\n{len(flagged)} / {len(rows)} clips flagged")
    tally = {}
    for r in flagged:
        for f in r["flags"].split("|"):
            tally[f] = tally.get(f, 0) + 1
    for f, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {f:24s} {n}")

    if flagged:
        print("\nWorst 15 (watch these in overlay.py first):")
        for r in sorted(flagged, key=lambda r: r["dominant_rate"])[:15]:
            print(f"  {r['clip']:40s} dom={r['dominant_rate']:.2f} "
                  f"pose={r['pose_rate']:.2f} [{r['flags']}]")

    if len(schemas) > 1 or len(mps) > 1:
        print("\n*** CONSISTENCY PROBLEM ***")
        print(f"  schema versions present: {sorted(schemas)}")
        print(f"  mediapipe versions present: {sorted(mps)}")
        print("  Your set was extracted with more than one configuration.")
        print("  Re-extract everything with one pinned setup before training.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
