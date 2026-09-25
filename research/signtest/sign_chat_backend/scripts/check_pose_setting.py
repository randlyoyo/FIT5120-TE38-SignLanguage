#!/usr/bin/env python3
"""Does the cuDNN setting of pose extraction change what Uni-Sign translates?

    SIGNCHAT_CONFIG=/content/signchat.yaml python scripts/check_pose_setting.py \\
        --clips /content/test_clips --manifest <Drive>/auslan_work/manifest.jsonl

Loads the recogniser the server uses (Uni-Sign + rtmlib, ~3 GB of GPU memory, next to a running
server) and translates every clip under each setting, EXHAUSTIVE (onnxruntime's default) first.
Clips are named <uid>.mp4, as the notebook's test-clip cell saves them; with --manifest the
reference sentence is shown too.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--algos", default="EXHAUSTIVE,HEURISTIC")
    args = ap.parse_args()
    import torch
    from signchat.config import load_config
    from signchat.sign2text import SignRecognizer

    clips = sorted(glob.glob(os.path.join(args.clips, "*.mp4")))
    refs = {}
    if args.manifest:
        with open(args.manifest) as fh:
            refs = {r["uid"]: r["text"] for r in map(json.loads, fh)}
    rec = SignRecognizer(load_config(), torch.device("cuda"))
    algos = args.algos.split(",")
    out = {}
    for algo in algos:
        rec.set_cudnn_algo(algo)
        rec.recognise(clips[0])                          # warm-up: first calls pay for new shapes
        t0 = time.time()
        out[algo] = [rec.recognise(c) for c in clips]
        print(f"{algo}: {len(clips)} clips in {time.time() - t0:.1f}s, "
              f"pose {sum(r['timings']['pose'] for r in out[algo]):.1f}s", flush=True)
    same = 0
    for i, c in enumerate(clips):
        uid = os.path.basename(c)[:-4]
        texts = [out[a][i]["text"] for a in algos]
        same += len(set(texts)) == 1
        print(f"\n{uid}" + (f"\n   reference : {refs[uid]}" if uid in refs else ""))
        for a, t in zip(algos, texts):
            print(f"   {a:<11}: {t}")
    print(f"\nidentical translations: {same}/{len(clips)}")


if __name__ == "__main__":
    main()
