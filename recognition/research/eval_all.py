#!/usr/bin/env python3
"""Score one trained encoder on every evaluation split.

Reuses metric_train.evaluate rather than reimplementing it, so the threshold
rule is the one the 2D numbers were produced under: tau is fixed on Valid and
then applied unchanged to every test split. Recomputing tau per split would
flatter every arm equally and answer a question nobody asked.
"""
import argparse, json, os
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--enc", type=Path, required=True)
ap.add_argument("--cache", default="cache3d")
ap.add_argument("--ndim", type=int, default=3)
ap.add_argument("--T", type=int, default=96)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

os.environ["SIGN_CACHE"] = a.cache
os.environ["SIGN_NDIM"] = str(a.ndim)
os.environ["SIGN_SPLITS"] = "Valid,Test_STU,Test_ITW,Test_TED,Test_SYN,Test_MTV"
import metric_data as M, metric_train as MT

(Xtr, ytr), evals, words = M.build_fixed(a.T)
dev = MT.device()
model = MT.Encoder(d_in=Xtr.shape[2], hidden=256, emb=256, depth=3).to(dev)
model.load_state_dict(torch.load(a.enc, map_location=dev))
evals_t = {k: (torch.from_numpy(x), y) for k, (x, y) in evals.items()}
res = MT.evaluate(model, evals_t, (torch.from_numpy(Xtr), ytr), words, dev)
for k, v in res.items():
    v["n"] = int(len(evals[k][1]))
json.dump(res, open(a.out, "w"), indent=2)
print(json.dumps(res, indent=2))
