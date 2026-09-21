#!/usr/bin/env python3
"""Score a trained encoder on Test_MTV, restricted to a common clip set.

    python mtv_eval.py --enc runs/final/encoder.pt --cache cache  --ndim 2
    python mtv_eval.py --enc runs/mtv3d/encoder.pt --cache cache3d --ndim 3

Test_MTV never entered training or model selection for either arm, and the 3D
extraction of it is still in flight, so the two arms only compare if they are
asked about the SAME clips. --stems pins that set; without it the intersection
of what both caches hold is used.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", type=Path, required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--ndim", type=int, default=2)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--stems", type=Path, default=None,
                    help="npy of stems to restrict Test_MTV to")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.environ["SIGN_CACHE"] = a.cache
    os.environ["SIGN_NDIM"] = str(a.ndim)
    os.environ["SIGN_SPLITS"] = "Valid,Test_MTV"
    import metric_data as M
    import metric_train as MT
    import dtw_eval as E

    (Xtr, ytr), evals, words = M.build_fixed(a.T)
    dev = MT.device()
    model = MT.Encoder(d_in=Xtr.shape[2], hidden=256, emb=256, depth=3).to(dev)
    model.load_state_dict(torch.load(a.enc, map_location=dev))

    # Recover the stems of the Test_MTV rows build_fixed kept, in its order.
    X_all, stems_all = M.load_fixed("Test_MTV", a.T)
    lab = M.load_labels("Test_MTV")
    widx = {w: i for i, w in enumerate(words)}
    keep = [i for i, s in enumerate(stems_all) if lab.get(M.label_key(s)) in widx]
    stems = [stems_all[i] for i in keep]

    Xq, yq = evals["Test_MTV"]
    if a.stems is not None:
        want = set(np.load(a.stems).tolist())
        sel = [i for i, s in enumerate(stems) if s in want]
        Xq, yq = Xq[sel], yq[sel]
        stems = [stems[i] for i in sel]

    tr_emb = MT.embed(model, torch.from_numpy(Xtr), dev=dev)
    D = MT.distance_matrix(tr_emb, np.asarray(ytr),
                           MT.embed(model, torch.from_numpy(Xq), dev=dev),
                           len(words))
    truth = np.asarray(yq)
    s, y = E.trials(D, truth, 50)
    auc, eer, tau, _ = E.metrics(s, y)
    out = dict(tag=a.tag, cache=a.cache, ndim=a.ndim, n=len(truth),
               auc=float(auc), eer=float(eer),
               rank1=float((D.argmin(1) == truth).mean()),
               rank5=float(np.mean([t in r for t, r in
                                    zip(truth, np.argsort(D, 1)[:, :5])])))
    print(json.dumps(out, indent=2))
    np.save(f"scores/mtv_stems_{a.tag or a.cache}.npy", np.array(stems))
    return out


if __name__ == "__main__":
    main()
