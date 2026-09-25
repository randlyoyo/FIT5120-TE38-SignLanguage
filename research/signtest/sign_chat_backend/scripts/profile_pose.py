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


def iou(a, b):
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def compare(outs, ref, videos, spec):
    """How much a setting changes what Uni-Sign sees. Low-confidence keypoints are zeroed by
    spec.load_part_kp before the model, so their pixel positions do not matter; confident ones,
    threshold crossings, the chosen person and the model input itself do."""
    conf, allpx, flips, n_kp, boxes, inp = [], [], 0, 0, 0, []
    for (kp, sc, au), (kr, sr, ar), (_, w, h, _) in zip(outs, ref, videos):
        px = np.linalg.norm((kp - kr) * [w, h], axis=-1)
        both = (sc >= spec.CONF_THRESHOLD) & (sr >= spec.CONF_THRESHOLD)
        conf.append(px[both]); allpx.append(px.ravel())
        flips += int(((sc >= spec.CONF_THRESHOLD) != (sr >= spec.CONF_THRESHOLD)).sum()); n_kp += sc.size
        boxes += int((iou(au["chosen_box"], ar["chosen_box"]) < 0.9).sum())
        a, b = spec.load_part_kp(kp, sc), spec.load_part_kp(kr, sr)
        inp.append(np.concatenate([np.abs(a[k][..., :2] - b[k][..., :2]).ravel() for k in spec.PART_ORDER]))
    conf, allpx, inp = np.concatenate(conf), np.concatenate(allpx), np.concatenate(inp)
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else 0.0
    return {"conf_p50_px": q(conf, 50), "conf_p99_px": q(conf, 99), "conf_max_px": float(conf.max()) if len(conf) else 0.0,
            "all_max_px": float(allpx.max()), "gate_flips": flips / max(n_kp, 1), "box_changed": boxes,
            "input_mean": float(inp.mean()), "input_p99": q(inp, 99), "input_max": float(inp.max())}


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
    import spec

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
            outs, t0 = [], time.perf_counter()
            for frames, w, h, _ in videos:
                outs.append(ex.run(frames, w, h))
            total = time.perf_counter() - t0
            if ref is None:
                ref = outs
            cmp = compare(outs, ref, videos, spec)
            diff = cmp["all_max_px"]
            sizes = collections.Counter(s for s, _ in shapes)
            first_new = [dt for s, dt in shapes if sizes[s] == 1]
            rows.append((algo, p, provider, total, store, diff, len(sizes), first_new))
            ms = 1000 * total / n_frames
            print(f"\n{algo} pass {p} ({provider}): {total:.2f}s total, {ms:.1f} ms/frame")
            print(f"   vs {rows[0][0]} pass 1 - confident keypoints (score >= {spec.CONF_THRESHOLD} in both): "
                  f"median {cmp['conf_p50_px']:.2f} px, p99 {cmp['conf_p99_px']:.2f} px, max {cmp['conf_max_px']:.2f} px")
            print(f"      keypoints crossing the {spec.CONF_THRESHOLD} threshold: {cmp['gate_flips']:.3%} | "
                  f"frames whose chosen person box moved (IoU < 0.9): {cmp['box_changed']}/{n_frames} | "
                  f"all keypoints incl. low-confidence: max {cmp['all_max_px']:.1f} px")
            print(f"      Uni-Sign model input (spec.load_part_kp, normalised units): "
                  f"mean |diff| {cmp['input_mean']:.4f}, p99 {cmp['input_p99']:.4f}, max {cmp['input_max']:.4f}")
            other = total - sum(store.values())
            for k in PARTS:
                print(f"   {k:<10}{store[k]:7.2f}s  {1000 * store[k] / n_frames:6.1f} ms/frame")
            print(f"   {'other':<10}{other:7.2f}s  {1000 * other / n_frames:6.1f} ms/frame")
            print(f"   pose batches: {len(shapes)} calls, {len(sizes)} sizes"
                  + (f"; one-off sizes took {np.mean(first_new):.2f}s each" if first_new else ""))
    print("\nsummary (ms/frame): " + " | ".join(f"{a} p{p}: {1000 * t / n_frames:.1f}" for a, p, _, t, *_ in rows))


if __name__ == "__main__":
    main()
