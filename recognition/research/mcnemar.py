#!/usr/bin/env python3
"""Paired comparison of arm A (2D) and arm C (3D) on Test_MTV.

A 3-point gap on 12820 clips is easy to over- or under-read from the marginal
rates alone. The two arms answer the same queries, so the right test is a
paired one on the clips where they DISAGREE -- McNemar -- not a two-sample
test that pretends the query sets are independent.
"""
import os
import numpy as np
import torch


def predictions(cache, ndim, enc, split="Test_MTV", T=96):
    os.environ["SIGN_CACHE"] = cache
    os.environ["SIGN_NDIM"] = str(ndim)
    os.environ["SIGN_SPLITS"] = f"Valid,{split}"
    for m in ("metric_data", "metric_train"):
        import sys
        sys.modules.pop(m, None)
    import metric_data as M, metric_train as MT
    (Xtr, ytr), evals, words = M.build_fixed(T)
    dev = MT.device()
    model = MT.Encoder(d_in=Xtr.shape[2], hidden=256, emb=256, depth=3).to(dev)
    model.load_state_dict(torch.load(enc, map_location=dev))
    X_all, stems_all = M.load_fixed(split, T)
    lab, widx = M.load_labels(split), {w: i for i, w in enumerate(words)}
    stems = [s for s in stems_all if lab.get(M.label_key(s)) in widx]
    Xq, yq = evals[split]
    tr_emb = MT.embed(model, torch.from_numpy(Xtr), dev=dev)
    D = MT.distance_matrix(tr_emb, np.asarray(ytr),
                           MT.embed(model, torch.from_numpy(Xq), dev=dev), len(words))
    return dict(zip(stems, (D.argmin(1) == np.asarray(yq))))


import sys
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "Test_MTV"
a = predictions("cache", 2, "runs/final/encoder.pt", SPLIT)
c = predictions("cache3d", 3, "runs/mtv3d/encoder.pt", SPLIT)
common = sorted(set(a) & set(c))
A = np.array([a[s] for s in common]); C = np.array([c[s] for s in common])
b, d = int((~A & C).sum()), int((A & ~C).sum())      # 3D-only right, 2D-only right
n = len(common)
chi2 = (abs(b - d) - 1) ** 2 / (b + d)
from math import erfc, sqrt
p = erfc(sqrt(chi2 / 2))
print(f"[{SPLIT}] 共同 clip {n}  (2D {len(a)}, 3D {len(c)})")
print(f"两者都对 {int((A&C).sum())}   都错 {int((~A&~C).sum())}")
print(f"仅 3D 对 {b}   仅 2D 对 {d}   净增 {b-d}")
print(f"rank-1: 2D {A.mean()*100:.2f}%  3D {C.mean()*100:.2f}%  差 {(C.mean()-A.mean())*100:+.2f}pt")
print(f"McNemar chi2={chi2:.1f}  p={p:.2e}")
