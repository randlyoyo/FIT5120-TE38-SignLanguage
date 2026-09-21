#!/usr/bin/env python3
"""Check that batched pose extraction reproduces the per-frame path.

`extract_pose.py --pose-batch N` changes one thing: how many crops go through
the pose model per call. This runs both paths on the same decoded frames.

What "the same" can mean
------------------------
Bit-identity is the wrong bar. RTMW decodes each keypoint as the argmax of a
SimCC distribution, and a batch-of-32 kernel computes the logits with ~1e-6
different rounding than a batch-of-1 kernel. Wherever the top two bins are tied
to within that noise, the argmax can land on the neighbour: the keypoint moves
by exactly one bin. Measured on news 78_191: 1 of 12,103 (crop, keypoint) pairs
flipped, with a top-two margin of 1.2e-7, by one bin (1.46 px), on a person who
was not selected, on a keypoint the model does not use. Any two numerical
implementations -- two GPUs, two ORT versions -- disagree this way; on a GPU the
rounding differences are larger, so there will be more of these, not fewer.

So a case PASSES when all three hold:

  structure    every frame yields the same number of people, in the same order
               -- a crop landing on the wrong frame, a broken "nobody detected"
               fallback, a mis-grouped batch all fail here
  model input  the four part tensors spec.load_part_kp builds for the selected
               signer agree to 1e-5. This is the data training actually sees,
               and it is strict: one bin on any confident keypoint of the
               selected person exceeds it
  flip rate    keypoints that moved at all are at most 0.1% of all candidate
               keypoints. A real bug moves many at once; tie flips are rare

Synthetic cases cover what real Signer-Only footage rarely contains but a
batching bug breaks first: blank frames (the detector finds nobody, so rtmlib
falls back to the whole image as the box) and two people side by side (one
frame yields several crops). Batch sizes that do not divide the crop count are
used so the last partial batch is exercised.

On Colab, verify the GPU path -- the one production uses -- straight from the
manifest:

    python verify_batching.py --from-manifest "$MANIFEST" --n 6 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402
from extract_pose import PoseExtractor, _read_frames, _VideoSource  # noqa: E402

PX_TOL = 0.01          # a candidate keypoint counts as "moved" beyond this
MODEL_IN_TOL = 1e-5    # model-input agreement required for the selected signer
FLIP_FRAC_MAX = 1e-3   # moved keypoints allowed, as a fraction of all


def synthetic_cases(frames):
    base = frames[: min(12, len(frames))]
    blank = [np.zeros_like(base[0])] * 3
    return {
        "synthetic:blank-frames": blank + base[:6] + blank,
        "synthetic:two-people": [np.hstack([f, f]) for f in base[:8]],
    }


def compare(ex, frames, batch, ref):
    """Batched candidates vs a precomputed per-frame reference, same frames."""
    ex.pose_batch = batch
    t = time.time()
    got = list(ex._candidates_batched(frames))
    t_got = time.time() - t

    moved = total = 0
    worst_px = 0.0
    persons = []
    for (rk, _, rb), (gk, _, gb) in zip(ref, got):
        rk, gk = np.asarray(rk, np.float64), np.asarray(gk, np.float64)
        if rk.ndim == 2:
            rk = rk[None]
        if rk.shape != gk.shape:
            return {"ok": False,
                    "why": f"person count differs {rk.shape} vs {gk.shape}"}
        if not np.array_equal(np.asarray(rb), np.asarray(gb)):
            return {"ok": False, "why": "detector boxes differ between paths"}
        persons.append(rk.shape[0])
        d = np.linalg.norm(rk - gk, axis=-1)           # (persons, 133)
        moved += int((d > PX_TOL).sum()); total += d.size
        worst_px = max(worst_px, float(d.max()))

    # The data training sees, built by the SAME selection code from each path.
    h, w = frames[0].shape[:2]
    kp1, sc1, _ = ex._assemble(ref, len(frames), w, h)
    kp2, sc2, _ = ex._assemble(got, len(frames), w, h)
    p1, p2 = spec.load_part_kp(kp1, sc1), spec.load_part_kp(kp2, sc2)
    model_in = max(float(np.abs(p1[k] - p2[k]).max()) for k in spec.PART_ORDER)

    frac = moved / max(total, 1)
    ok = model_in <= MODEL_IN_TOL and frac <= FLIP_FRAC_MAX
    return {"ok": ok, "frames": len(frames), "crops": int(sum(persons)),
            "max_persons": int(max(persons)), "moved": moved, "frac": frac,
            "px": worst_px, "model_input": model_in, "t_batch": t_got}


def clips_from_manifest(path, n, seed, tmpdir):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        with _VideoSource(r["video"], tmpdir) as src:
            dst = tmpdir / f"{r['uid']}.mp4"
            shutil.copy(src.path, dst)
            out.append(str(dst))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clips", nargs="*", help="video files or directories")
    ap.add_argument("--from-manifest", default=None,
                    help="sample clips from a manifest instead (reads zip refs)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batches", type=int, nargs="+", default=[7, 32])
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    tmpdir = Path(tempfile.mkdtemp(prefix="verify-batching-"))
    try:
        files = []
        for c in args.clips:
            files += (sorted(glob.glob(os.path.join(c, "*.mp4")))
                      if os.path.isdir(c) else [c])
        if args.from_manifest:
            files += clips_from_manifest(args.from_manifest, args.n, args.seed, tmpdir)
        if not files:
            ap.error("give clip paths or --from-manifest")

        ex = PoseExtractor(device=args.device)
        cases = {}
        for f in files:
            frames, *_ = _read_frames(Path(f), args.max_frames)
            cases[os.path.basename(f)] = frames
        cases.update(synthetic_cases(next(iter(cases.values()))))

        print(f"device={args.device}\n")
        print(f"{'case':38} {'batch':>5} {'frames':>6} {'crops':>5} {'maxP':>4} "
              f"{'moved':>5} {'moved%':>7} {'max px':>7} {'model in':>9}  result")
        failures = 0
        t_ref_total = t_batch_total = 0.0
        for name, frames in cases.items():
            ex.pose_batch = 1
            t = time.time()
            ref = list(ex._candidates_per_frame(frames))   # upstream, once
            t_ref = time.time() - t
            for b in args.batches:
                r = compare(ex, frames, b, ref)
                if "why" in r:
                    print(f"{name[:38]:38} {b:5d}  FAIL: {r['why']}")
                    failures += 1
                    continue
                failures += not r["ok"]
                if b == max(args.batches):
                    t_ref_total += t_ref; t_batch_total += r["t_batch"]
                print(f"{name[:38]:38} {b:5d} {r['frames']:6d} {r['crops']:5d} "
                      f"{r['max_persons']:4d} {r['moved']:5d} {r['frac']:7.3%} "
                      f"{r['px']:7.2f} {r['model_input']:9.2e}  "
                      f"{'PASS' if r['ok'] else 'FAIL'}")

        print(f"\ncriteria: model input <= {MODEL_IN_TOL}, moved keypoints <= "
              f"{FLIP_FRAC_MAX:.1%}, same people per frame")
        print(f"pose timing at batch {max(args.batches)} on {args.device}: "
              f"per-frame {t_ref_total:.1f}s vs batched {t_batch_total:.1f}s "
              f"({t_ref_total / max(t_batch_total, 1e-9):.2f}x)")
        print("ALL PASS" if failures == 0 else f"{failures} FAILED")
        return 1 if failures else 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
