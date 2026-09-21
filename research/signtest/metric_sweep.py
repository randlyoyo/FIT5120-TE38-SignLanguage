#!/usr/bin/env python3
"""Random hyper-parameter search, resumable, one JSON row per trial.

    python metric_sweep.py --trials 20 --epochs 30
    python metric_sweep.py --final                 # retrain the best, long

Random rather than grid: with a dozen knobs a grid wastes almost every trial
varying something that does not matter, while random search spends its budget
across all of them at once.

Trials are ranked on Valid AUC only. The test splits are never consulted during
the search -- picking a configuration by test score is the same mistake as
picking a threshold on it.

Test_MTV is excluded throughout. It is a multi-view capture and 2D keypoints
cannot bridge the viewpoint change, so including it would only add noise to
model selection.
"""

import argparse
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import metric_train as T

SPACE = dict(
    lr=lambda r: 10 ** r.uniform(-3.4, -2.2),
    hidden=lambda r: r.choice([192, 256, 384]),
    emb=lambda r: r.choice([128, 256, 384]),
    depth=lambda r: r.choice([3, 4, 5]),
    drop_p=lambda r: r.choice([0.0, 0.1, 0.2, 0.3]),
    wd=lambda r: r.choice([0.01, 0.05, 0.1]),
    bs=lambda r: r.choice([128, 256]),
    T=lambda r: r.choice([48, 64, 96]),
    margin=lambda r: r.choice([0.2, 0.3, 0.4, 0.5]),
    scale_s=lambda r: r.choice([20.0, 30.0, 45.0]),
    ls=lambda r: r.choice([0.0, 0.1, 0.2]),
    # augmentation -- the highest-leverage group here, since 12 examples per
    # class is little enough that a 1.3M-parameter encoder can memorise them
    speed=lambda r: r.choice([0.0, 0.1, 0.2, 0.3]),
    scale=lambda r: r.choice([0.0, 0.05, 0.1, 0.2]),
    rot=lambda r: r.choice([0.0, 5.0, 10.0, 15.0]),
    shift=lambda r: r.choice([0.0, 0.03, 0.05, 0.1]),
    drop=lambda r: r.choice([0.0, 0.05, 0.1, 0.2]),
    noise=lambda r: r.choice([0.0, 0.005, 0.01, 0.02]),
)


def sample(r):
    cfg = dict(T.DEFAULT)
    for k, f in SPACE.items():
        v = f(r)
        cfg[k] = float(v) if isinstance(v, (float, np.floating)) else int(v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("runs/sweep"))
    ap.add_argument("--final", action="store_true",
                    help="retrain the best config for --final-epochs")
    ap.add_argument("--final-epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "trials.jsonl"

    done = []
    if log.exists():
        done = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    print(f"device={T.device()}  {len(done)} trial(s) already logged")

    if args.final:
        if not done:
            raise SystemExit("no trials to pick from")
        best = max(done, key=lambda d: d["val_auc"])
        cfg = dict(best["cfg"])
        cfg["epochs"] = args.final_epochs
        cfg["eval_every"] = 5
        print(f"final run from trial {best['trial']} "
              f"(search AUC {best['val_auc']:.4f}), {args.final_epochs} epochs")
        d = args.out.parent / "final"
        model, evals, train_t, words, hist, bv = T.train(
            cfg, ckpt=d / "checkpoint.pt")
        res = T.evaluate(model, evals, train_t, words, T.device())
        d.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), d / "encoder.pt")
        json.dump(dict(cfg=cfg, best_val_auc=bv, history=hist, results=res),
                  open(d / "result.json", "w"), indent=2)
        print(f"\n{'split':10s} {'AUC':>7s} {'EER':>7s} {'rank-1':>7s}")
        for k, v in res.items():
            print(f"{k:10s} {v['auc']:7.4f} {v['eer']*100:6.2f}% "
                  f"{v['rank1']*100:6.2f}%")
        return

    r = random.Random(args.seed + len(done))
    for i in range(len(done), len(done) + args.trials):
        cfg = sample(r)
        cfg["epochs"] = args.epochs
        cfg["eval_every"] = max(5, args.epochs // 4)
        cfg["seed"] = args.seed
        t0 = time.time()
        print(f"\n=== trial {i} === " + " ".join(
            f"{k}={cfg[k]}" for k in SPACE), flush=True)
        try:
            _, _, _, _, hist, val = T.train(cfg, verbose=False)
            row = dict(trial=i, val_auc=float(val), cfg=cfg,
                       minutes=(time.time() - t0) / 60,
                       loss=hist[-1]["loss"] if hist else None)
        except Exception as e:                              # noqa: BLE001
            # One bad config (an MPS op gap, an OOM) must not end the sweep.
            traceback.print_exc()
            row = dict(trial=i, val_auc=-1.0, cfg=cfg, error=str(e),
                       minutes=(time.time() - t0) / 60)
        with open(log, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        done.append(row)
        best = max(done, key=lambda d: d["val_auc"])
        print(f"  -> AUC {row['val_auc']:.4f}  ({row['minutes']:.1f} min)"
              f"   best so far: trial {best['trial']} {best['val_auc']:.4f}",
              flush=True)

    print(f"\n{'trial':>6s} {'val AUC':>8s}  top configs")
    for d in sorted(done, key=lambda d: -d["val_auc"])[:8]:
        keys = ["lr", "hidden", "emb", "depth", "T", "margin", "drop_p",
                "speed", "rot", "drop"]
        print(f"{d['trial']:6d} {d['val_auc']:8.4f}  " +
              " ".join(f"{k}={d['cfg'][k]}" for k in keys))


if __name__ == "__main__":
    main()
