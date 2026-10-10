#!/usr/bin/env python3
"""Faster signing: fewer ODE steps, and guidance in one batch. How much faster, and does it change the signing?

    SIGNCHAT_CONFIG=/content/signchat.yaml python scripts/check_speed.py --test-lmdb <local copy of lmdb_smooth> --steps 12 10 8

Uses the server's own config (bf16 weights, parallel streams), with the server stopped (GPU memory).

1. guidance in one batch (config batch_guidance): at 20 steps it must give the two-pass result, up to
   GPU rounding: the joints are compared with the two-pass ones (should be well under a millimetre).
2. fewer steps (config steps): judged like scripts/check_bf16.py, against the model's own seed-to-seed
   variation. A step count passes when, for every stream and test metric (signspark_ft.evaluate on
   the first --n test clips: txt_dtw, kf_err, txt_motion), it moves the value no more than a change of
   seed does (or 1%), and on everyday sentences its joints move no more than a seed change moves them.
3. speed: seconds per sentence through the server's features() for each setting (after a warm-up).

Prints a table, the smallest step count that passes, and writes --out. Exit code 0 when the batched
guidance matches; the step verdict is in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from check_bf16 import METRICS, SENTENCES, joint_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-lmdb", required=True, help="folder holding test/AuslanDaily_test.lmdb (lmdb_smooth)")
    ap.add_argument("--steps", type=int, nargs="+", default=[12, 10, 8], help="step counts to try (reference: 20)")
    ap.add_argument("--n", type=int, default=128, help="test clips per stream")
    ap.add_argument("--repeats", type=int, default=2, help="timing runs per sentence")
    ap.add_argument("--out", default="/content/check_speed.json")
    args = ap.parse_args()
    import torch
    from signchat.config import load_config
    from signchat.text2sign import STREAMS, SignGenerator

    cfg = load_config()
    REF = 20                                             # the setting every reported result used
    gen = SignGenerator(cfg, torch.device("cuda"))
    ft, R, dev, c = gen.ft, gen.R, gen.device, gen.cfg
    batches = {s: ft.eval_batches(s, ft.lmdb_path(args.test_lmdb, "test"), args.n) for s in STREAMS}
    groups = joint_groups(R)

    def setting(steps, batched, seed=0):
        c["steps"], c["batch_guidance"], c["seed"] = steps, batched, seed

    def sentence_joints():
        return {t: gen.smplx(gen.features(t))[1] for t in SENTENCES}

    def seconds_per_sentence():
        gen.features(SENTENCES[0])                       # warm-up for this setting
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.repeats):
            for t in SENTENCES:
                gen.features(t)
        return round((time.time() - t0) / (args.repeats * len(SENTENCES)), 3)

    def test_metrics(steps, seed):
        return {s: ft.evaluate(gen.models[s][0], gen.models[s][1], batches[s], s, gen.amp, dev,
                               kf_steps=steps, txt_steps=steps, seed=seed) for s in STREAMS}

    def joint_diff(A, B):
        out = {}
        for g, idx in groups.items():
            d = [np.linalg.norm(A[t][:, idx] - B[t][:, idx], axis=-1) * 1000 for t in SENTENCES]
            out[g] = {"mean_mm": round(float(np.mean([x.mean() for x in d])), 2),
                      "max_mm": round(float(max(x.max() for x in d)), 1)}
        return out

    t_start = time.time()
    res = {"n_test_clips": args.n, "reference_steps": REF, "sentences": SENTENCES, "seconds_per_sentence": {}}

    # reference: REF steps, two passes per guided step; seeds 0 and 1
    setting(REF, False, 1)
    J1 = sentence_joints()
    setting(REF, False, 0)
    J0 = sentence_joints()
    res["seconds_per_sentence"][f"{REF} steps, two passes"] = seconds_per_sentence()
    res["test_seed0"], res["test_seed1"] = test_metrics(REF, 0), test_metrics(REF, 1)
    res["joints_seed1_vs_seed0"] = joint_diff(J1, J0)

    # 1. guidance in one batch, same steps: should be the same signing
    setting(REF, True, 0)
    JB = sentence_joints()
    res["seconds_per_sentence"][f"{REF} steps, one batch"] = seconds_per_sentence()
    res["joints_batched_vs_two_pass"] = joint_diff(JB, J0)
    batched_ok = all(v["max_mm"] < 1.0 for v in res["joints_batched_vs_two_pass"].values())

    # 2. fewer steps (with guidance in one batch, as the server would run them)
    rows, passing = [], []
    for n in args.steps:
        setting(n, True, 0)
        Jn = sentence_joints()
        res["seconds_per_sentence"][f"{n} steps, one batch"] = seconds_per_sentence()
        tm = test_metrics(n, 0)
        jd = joint_diff(Jn, J0)
        res[f"test_{n}_steps"], res[f"joints_{n}_steps_vs_reference"] = tm, jd
        ok = True
        for s in STREAMS:
            for m in METRICS:
                a, b, x = res["test_seed0"][s][m], res["test_seed1"][s][m], tm[s][m]
                tol = max(abs(b - a), 0.01 * abs(a))
                good = abs(x - a) <= tol
                ok &= good
                rows.append(f"  {n:>2} steps  {s:<5} {m:<11} ref {a:9.4f} | seed change {b - a:+.4f} | "
                            f"step change {x - a:+.4f}  {'ok' if good else 'LARGER THAN SEED CHANGE'}")
        for g in groups:
            good = jd[g]["mean_mm"] <= res["joints_seed1_vs_seed0"][g]["mean_mm"]
            ok &= good
            rows.append(f"  {n:>2} steps  joints {g:<12} vs reference: mean {jd[g]['mean_mm']:.2f} mm "
                        f"(max {jd[g]['max_mm']:.1f}) | seed change: mean {res['joints_seed1_vs_seed0'][g]['mean_mm']:.2f} mm"
                        f"  {'ok' if good else 'LARGER THAN SEED CHANGE'}")
        res[f"pass_{n}_steps"] = bool(ok)
        if ok:
            passing.append(n)
    best = min(passing) if passing else REF
    res["recommended_steps"], res["batched_guidance_matches"] = best, batched_ok
    res["seconds"] = round(time.time() - t_start)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)

    print("seconds per sentence (signing only, features()):")
    for k, v in res["seconds_per_sentence"].items():
        print(f"  {k:<24} {v:.2f} s")
    print("\nguidance in one batch vs two passes, same steps (joint difference):")
    for g, v in res["joints_batched_vs_two_pass"].items():
        print(f"  {g:<12} mean {v['mean_mm']:.3f} mm, max {v['max_mm']:.2f} mm")
    print(f"  -> {'the same signing (GPU rounding only)' if batched_ok else 'DIFFERENT: keep batch_guidance false'}")
    print(f"\nfewer steps (reference {REF}, test clips per stream {args.n}):\n" + "\n".join(rows))
    for n in args.steps:
        print(f"  {n:>2} steps: {'PASSES' if res[f'pass_{n}_steps'] else 'fails'}")
    print(f"\nrecommended: steps = {best}" + ("" if passing else " (no smaller step count passed)") +
          f"; batch_guidance = {batched_ok}   ({res['seconds']}s; details in {args.out})")
    sys.exit(0 if batched_ok else 1)


if __name__ == "__main__":
    main()
