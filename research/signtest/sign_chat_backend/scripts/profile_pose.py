#!/usr/bin/env python3
"""Where pose extraction spends its time, and whether a cuDNN setting makes it faster without
changing the keypoints.

    python scripts/profile_pose.py --clips /content/test_clips

On an A100 pose extraction took ~40 ms per frame, far more than the two small ONNX models need.
This runs the exact extraction the server uses (extract_pose.PoseExtractor, batched pose model) on
the clips and splits the time into:

    det_pre / det_run / det_post     YOLOX person detector, once per frame, batch 1
    pose_pre / pose_run / pose_post  RTMW crops, the pose model in batches of --batch
    other                            person selection, bookkeeping

for three settings of onnxruntime's cudnn_conv_algo_search:
    EXHAUSTIVE  onnxruntime's default (what the server runs now): benchmarks every convolution
                algorithm the first time it sees an input shape - and the last pose batch of every
                clip has a new size
    HEURISTIC   cuDNN picks from its heuristics, no benchmarking
    DEFAULT     the default algorithm

Each setting runs the clips twice: pass 1 pays for new shapes, pass 2 shows the steady state.
Keypoints of every setting are compared with EXHAUSTIVE pass 1 (max difference in pixels).
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = ("det_pre", "det_run", "det_post", "pose_pre", "pose_run", "pose_post")


def timed(store, name, fn, shapes=None):
    def wrapper(*a, **kw):
        t = time.perf_counter()
        out = fn(*a, **kw)
        dt = time.perf_counter() - t
        store[name] += dt
        if shapes is not None:                       # pose batches: (batch size, seconds)
            feed = a[1] if len(a) > 1 else kw.get("input_feed")
            shapes.append((next(iter(feed.values())).shape[0], dt))
        return out
    return wrapper


def rebuild_sessions(wb, algo):
    import onnxruntime as ort
    opts = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": algo}), "CPUExecutionProvider"]
    for m in (wb.det_model, wb.pose_model):
        m.session = ort.InferenceSession(m.session._model_path, providers=opts)
    return wb.pose_model.session.get_providers()[0]


def instrument(wb, store, shapes):
    det, pose = wb.det_model, wb.pose_model
    for m, pre in ((det, "det"), (pose, "pose")):
        for fn in ("preprocess", "postprocess"):
            orig = getattr(type(m), fn).__get__(m)          # the class method, so re-instrumenting never nests
            setattr(m, fn, timed(store, f"{pre}_{fn[:4] if fn == 'postprocess' else 'pre'}", orig))
    det.inference = timed(store, "det_run", type(det).inference.__get__(det))
    run = type(pose.session).run.__get__(pose.session)
    pose.session.run = timed(store, "pose_run", run, shapes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="folder of .mp4 clips (e.g. the notebook's /content/test_clips)")
    ap.add_argument("--code", default=os.path.join(HERE, "..", "..", "unisign"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--algos", default="EXHAUSTIVE,HEURISTIC,DEFAULT")
    ap.add_argument("--device", default="cuda", help="cpu: only checks the script (no cuDNN settings)")
    args = ap.parse_args()
    import onnxruntime as ort
    try:
        ort.preload_dlls()
    except Exception:
        pass
    sys.path.insert(0, os.path.abspath(args.code))
    import extract_pose

    clips = sorted(glob.glob(os.path.join(args.clips, "*.mp4")))
    videos = [extract_pose._read_frames(c) for c in clips]
    n_frames = sum(len(v[0]) for v in videos)
    print(f"onnxruntime {ort.__version__} | {len(clips)} clips, {n_frames} frames | pose batch {args.batch}")
    ex = extract_pose.PoseExtractor(device=args.device, pose_batch=args.batch)
    if args.device != "cuda":
        args.algos = "CPU"
    ref = None
    rows = []
    for algo in args.algos.split(","):
        provider = rebuild_sessions(ex._wholebody, algo) if algo != "CPU" else "CPUExecutionProvider"
        for p in (1, 2):
            store, shapes = collections.defaultdict(float), []
            instrument(ex._wholebody, store, shapes)
            kps, t0 = [], time.perf_counter()
            for frames, w, h, _ in videos:
                kps.append(ex.run(frames, w, h)[0])
            total = time.perf_counter() - t0
            if ref is None:
                ref = kps
            diff = max(float(np.abs((a - b) * [v[1], v[2]]).max()) for a, b, v in zip(kps, ref, videos))
            sizes = collections.Counter(s for s, _ in shapes)
            first_new = [dt for s, dt in shapes if sizes[s] == 1]
            rows.append((algo, p, provider, total, store, diff, len(sizes), first_new))
            ms = 1000 * total / n_frames
            print(f"\n{algo} pass {p} ({provider}): {total:.2f}s total, {ms:.1f} ms/frame, "
                  f"keypoints max diff vs {rows[0][0]} pass 1: {diff:.3f} px")
            other = total - sum(store.values())
            for k in PARTS:
                print(f"   {k:<10}{store[k]:7.2f}s  {1000 * store[k] / n_frames:6.1f} ms/frame")
            print(f"   {'other':<10}{other:7.2f}s  {1000 * other / n_frames:6.1f} ms/frame")
            print(f"   pose batches: {len(shapes)} calls, {len(sizes)} sizes"
                  + (f"; one-off sizes took {np.mean(first_new):.2f}s each" if first_new else ""))
    print("\nsummary (ms/frame): " + " | ".join(f"{a} p{p}: {1000 * t / n_frames:.1f}" for a, p, _, t, *_ in rows))


if __name__ == "__main__":
    main()
