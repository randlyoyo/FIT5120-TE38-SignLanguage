#!/usr/bin/env python3
"""Can the three SignSparK generators be stored in bf16 (half the memory) without changing the signing?

    SIGNCHAT_CONFIG=/content/signchat.yaml python scripts/check_bf16.py --test-lmdb <Drive>/auslan_work/smplx_full/lmdb_smooth --n 256

The yardstick is the model's own seed-to-seed variation: generation starts from random noise, so
fp32 with seed 0 and fp32 with seed 1 already differ. bf16 passes when it moves the outputs no more
than a change of seed does.

1. Test-set metrics, as training chose its checkpoints (signspark_ft.evaluate on the first --n test
   clips of each stream): text-only DTW to the real signing (txt_dtw, degrees for hand/body), error
   between given keyframes (kf_err), motion relative to real signing (txt_motion). Three runs per
   stream: fp32 seed 0, fp32 seed 1, bf16 seed 0.
2. The server's own generation (features(): retrieval, keyframes, smoothing) on everyday sentences,
   through SMPL-X: joint position differences in mm, fp32 vs bf16 and fp32 seed 0 vs seed 1, for the
   fingers, the wrists / arms and the face.

Needs ~20 GB of GPU memory: run it with the server stopped. Exit code 1 if bf16 fails the rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SENTENCES = ["Hello, how are you?", "I am good, thank you.", "What is your name?", "See you tomorrow.",
             "I am hungry, let us eat something together.", "My family lives in Melbourne.",
             "Thank you very much.", "Where are you going?"]
METRICS = ("txt_dtw", "kf_err", "txt_motion")


def joint_groups(R):
    fingers = R.L_FINGERS + R.R_FINGERS
    arms = [16, 17, 18, 19, R.L_WRIST, R.R_WRIST]
    return {"fingers": fingers, "arms+wrists": arms, "face": list(R.FACE_LMK)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-lmdb", required=True, help="folder holding test/AuslanDaily_test.lmdb (lmdb_smooth)")
    ap.add_argument("--n", type=int, default=256, help="test clips per stream")
    ap.add_argument("--out", default="/content/check_bf16.json")
    args = ap.parse_args()
    import torch
    from signchat.config import load_config
    from signchat.text2sign import STREAMS, SignGenerator

    cfg = load_config(overrides={"text2sign": {"weights_dtype": "float32"}})
    gen = SignGenerator(cfg, torch.device("cuda"))
    ft, R, dev = gen.ft, gen.R, gen.device
    test_path = lambda s: ft.lmdb_path(args.test_lmdb, "test")
    batches = {s: ft.eval_batches(s, test_path(s), args.n) for s in STREAMS}
    groups = joint_groups(R)

    def test_metrics(seed):
        out = {}
        for s in STREAMS:
            model, flow = gen.models[s]
            out[s] = ft.evaluate(model, flow, batches[s], s, gen.amp, dev, seed=seed)
        return out

    def sentence_joints(seed):
        gen.cfg["seed"] = seed
        return {t: gen.smplx(gen.features(t, parallel=False))[1] for t in SENTENCES}

    def mem_gb():
        return round(sum(p.numel() * p.element_size() for s in STREAMS for n, p in gen.models[s][0].named_parameters()
                         if not n.startswith(ft.TEXT_KEY)) / 2**30, 2)

    t0 = time.time()
    res = {"n_test_clips": args.n, "generator_weights_gb": {"float32": mem_gb()}}
    res["fp32_seed0"], J0 = test_metrics(0), sentence_joints(0)
    res["fp32_seed1"], J1 = test_metrics(1), sentence_joints(1)
    gen._cast_generators(torch.bfloat16)
    res["generator_weights_gb"]["bfloat16"] = mem_gb()
    res["bf16_seed0"], JB = test_metrics(0), sentence_joints(0)
    gen.cfg["seed"] = 0

    def joint_diff(A, B):          # mean over frames of the per-group mean joint distance, mm; and the max
        out = {}
        for g, idx in groups.items():
            d = [np.linalg.norm(A[t][:, idx] - B[t][:, idx], axis=-1) * 1000 for t in SENTENCES]
            out[g] = {"mean_mm": round(float(np.mean([x.mean() for x in d])), 2),
                      "max_mm": round(float(max(x.max() for x in d)), 1)}
        return out
    res["joints_bf16_vs_fp32"] = joint_diff(JB, J0)
    res["joints_seed1_vs_seed0"] = joint_diff(J1, J0)

    # the rule: for every stream and metric, |bf16 - fp32| <= |seed1 - seed0| (or 1% of the value);
    # for every joint group, bf16's mean joint difference <= the seed change's
    verdict, rows = True, []
    for s in STREAMS:
        for m in METRICS:
            a, b, c = res["fp32_seed0"][s][m], res["fp32_seed1"][s][m], res["bf16_seed0"][s][m]
            tol = max(abs(b - a), 0.01 * abs(a))
            ok = abs(c - a) <= tol
            verdict &= ok
            rows.append(f"  {s:<5} {m:<11} fp32 s0 {a:9.4f} | fp32 s1 {b:9.4f} (seed change {b - a:+.4f}) | "
                        f"bf16 s0 {c:9.4f} (bf16 change {c - a:+.4f})  {'ok' if ok else 'LARGER THAN SEED CHANGE'}")
    for g in groups:
        ok = res["joints_bf16_vs_fp32"][g]["mean_mm"] <= res["joints_seed1_vs_seed0"][g]["mean_mm"]
        verdict &= ok
        rows.append(f"  joints {g:<12} bf16 vs fp32: mean {res['joints_bf16_vs_fp32'][g]['mean_mm']:.2f} mm "
                    f"(max {res['joints_bf16_vs_fp32'][g]['max_mm']:.1f}) | seed 1 vs 0: mean "
                    f"{res['joints_seed1_vs_seed0'][g]['mean_mm']:.2f} mm (max {res['joints_seed1_vs_seed0'][g]['max_mm']:.1f})"
                    f"  {'ok' if ok else 'LARGER THAN SEED CHANGE'}")
    res["pass"] = bool(verdict)
    res["seconds"] = round(time.time() - t0)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"generator weights: fp32 {res['generator_weights_gb']['float32']} GB -> bf16 {res['generator_weights_gb']['bfloat16']} GB")
    print(f"test clips per stream: {args.n}\n" + "\n".join(rows))
    print(f"\nbf16 {'PASSES' if verdict else 'FAILS'}: its effect is {'within' if verdict else 'beyond'} the seed-to-seed variation "
          f"({res['seconds']}s; details in {args.out})")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
