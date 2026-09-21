#!/usr/bin/env python3
"""Train a sequence encoder with ArcFace, then score it on the DTW protocol.

    python metric_train.py --epochs 100 --out runs/base

Why ArcFace rather than a triplet loss: with 3215 classes and 12 examples each,
triplet mining has terrible signal-to-noise and needs a sampler to babysit.
ArcFace is a classification head with an angular margin -- no mining, no pair
construction -- and the penultimate layer is the embedding once training ends.

Evaluation deliberately reuses dtw_eval: it consumes an (n_queries, n_words)
distance matrix and does not care whether the distance came from warping two
sequences or from a dot product. Same splits, same impostor sampling, same
threshold-fixed-on-Valid rule, so the numbers sit directly beside the DTW ones.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader

import dtw_eval as E
import metric_data as M


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Encoder(nn.Module):
    """(B,T,d_in) -> L2-normalised (B,emb).

    Dilated 1D convolutions rather than a transformer: the sequences are ~64
    frames and the training set is small, so a compact receptive-field stack
    generalises better than attention over a short sequence would.

    Attention pooling rather than mean pooling: a sign's identity lives in a
    few key frames (the hold, the handshape change) and averaging over the
    lead-in dilutes exactly those.
    """

    def __init__(self, d_in=98, hidden=256, emb=256, depth=4, drop=0.1):
        super().__init__()
        layers, c = [], d_in
        for i in range(depth):
            layers += [
                nn.Conv1d(c, hidden, 5, padding=2 * (2 ** i), dilation=2 ** i),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(drop),
            ]
            c = hidden
        self.net = nn.Sequential(*layers)
        self.att = nn.Conv1d(hidden, 1, 1)
        self.fc = nn.Linear(hidden * 2, emb)
        self.bn = nn.BatchNorm1d(emb)

    def forward(self, x):
        z = self.net(x.transpose(1, 2))                 # (B,H,T)
        w = torch.softmax(self.att(z), dim=-1)
        # attention-weighted mean, concatenated with a max over time: the mean
        # carries the overall trajectory, the max the sharpest moment
        pooled = torch.cat([(z * w).sum(-1), z.amax(-1)], dim=1)
        return Fn.normalize(self.bn(self.fc(pooled)), dim=1)


class ArcFace(nn.Module):
    def __init__(self, emb, n_cls, scale=30.0, margin=0.3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, emb) * 0.01)
        self.s, self.m = scale, margin

    def forward(self, z, y):
        cos = Fn.linear(z, Fn.normalize(self.W, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = Fn.one_hot(y, cos.shape[1]).float()
        return self.s * (onehot * target + (1 - onehot) * cos)


@torch.no_grad()
def embed(model, X, bs=1024, dev=None):
    """X is an (N,T,D) tensor already on device or in host memory."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(dev)).cpu())
    return torch.cat(out).numpy()


def distance_matrix(train_emb, train_lab, q_emb, n_words):
    """(n_q, n_words) cosine distance, min over each word's templates.

    Mirrors the DTW scorer's min-over-templates rule so the two systems answer
    the same question.
    """
    bank = [[] for _ in range(n_words)]
    for e, l in zip(train_emb, train_lab):
        bank[l].append(e)
    k = min(len(b) for b in bank)
    B = np.stack([np.stack(b[:k]) for b in bank])          # (W,k,dim)
    sim = q_emb @ B.reshape(-1, B.shape[-1]).T             # (n_q, W*k)
    return 1.0 - sim.reshape(len(q_emb), n_words, k).max(axis=2)


def evaluate(model, evals, train, words, dev, n_neg=50):
    Xtr, ytr = train
    tr_emb = embed(model, Xtr, dev=dev)
    tr_lab = np.asarray(ytr)
    res, tau = {}, None
    for sp in ["Valid"] + [s for s in evals if s != "Valid"]:
        if sp not in evals:
            continue
        Xq, yq = evals[sp]
        D = distance_matrix(tr_emb, tr_lab, embed(model, Xq, dev=dev), len(words))
        truth = np.asarray(yq)
        s, y = E.trials(D, truth, n_neg)
        auc, eer, t, _ = E.metrics(s, y)
        if sp == "Valid":
            tau = t
        far, frr = E.at_threshold(s, y, tau)
        res[sp] = dict(auc=float(auc), eer=float(eer), far=float(far),
                       frr=float(frr),
                       rank1=float((D.argmin(1) == truth).mean()))
    return res


def train(cfg, verbose=True, ckpt=None):
    """Train an encoder. `ckpt` enables resume: progress is written there after
    every evaluation, and an existing file is picked up rather than restarted.

    Without it an interrupted run loses everything -- the best weights live only
    in a local variable until the very end. A 120-epoch run is over an hour.
    """
    dev = device()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    aug = {k: cfg[k] for k in
           ("speed", "scale", "rot", "shift", "drop", "noise")}
    (Xtr, ytr), evals, words = M.build_fixed(cfg["T"])
    # The whole training set is ~1 GB at T=64; keeping it resident on device
    # turns the input pipeline into an index operation.
    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    evals_t = {k: (torch.from_numpy(a), b) for k, (a, b) in evals.items()}
    train_t = (torch.from_numpy(Xtr), ytr)

    # d_in from the data, not the default: the 3D features are 146-dim where
    # the 2D ones are 98, and a silently wrong d_in would be a shape error at
    # best and a truncated input at worst.
    model = Encoder(d_in=Xtr.shape[2], hidden=cfg["hidden"], emb=cfg["emb"],
                    depth=cfg["depth"], drop=cfg["drop_p"]).to(dev)
    head = ArcFace(cfg["emb"], len(words), cfg["scale_s"], cfg["margin"]).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=cfg["lr"], weight_decay=cfg["wd"])
    n, bs = len(Xtr_t), cfg["bs"]
    steps = n // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=steps,
        pct_start=0.15)

    best, hist, start_ep = None, [], 0
    if ckpt is not None and Path(ckpt).exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        if st.get("cfg") != cfg:
            # A checkpoint from different hyper-parameters would silently
            # produce a model that is neither configuration.
            raise SystemExit(f"{ckpt} was written with a different cfg; "
                             "delete it or point --ckpt elsewhere")
        model.load_state_dict(st["model"]); head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        best, hist, start_ep = (st["best_auc"], st["best_model"]), st["hist"], st["epoch"]
        if verbose:
            print(f"  resumed from epoch {start_ep} (best AUC {st['best_auc']:.4f})")

    def save(ep):
        if ckpt is None:
            return
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        tmp = str(ckpt) + ".tmp"
        # Write then rename: a crash mid-write must not destroy the last good
        # checkpoint, which is the whole point of having one.
        torch.save({"cfg": cfg, "epoch": ep, "hist": hist,
                    "model": model.state_dict(), "head": head.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best_auc": best[0], "best_model": best[1]}, tmp)
        os.replace(tmp, ckpt)

    for ep in range(start_ep, cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = seen = 0
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb = M.augment_gpu(Xtr_t[idx], aug)
            yb = ytr_t[idx]
            opt.zero_grad()
            loss = Fn.cross_entropy(head(model(xb), yb), yb,
                                    label_smoothing=cfg["ls"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            seen += len(yb)
        if (ep + 1) % cfg["eval_every"] == 0 or ep == cfg["epochs"] - 1:
            r = evaluate(model, {"Valid": evals_t["Valid"]}, train_t, words, dev)
            auc = r["Valid"]["auc"]
            hist.append(dict(epoch=ep + 1, loss=tot / seen, val_auc=auc))
            if best is None or auc > best[0]:
                best = (auc, {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()})
            if verbose:
                print(f"  ep {ep+1:3d}  loss {tot/seen:6.3f}  Valid AUC {auc:.4f}"
                      f"{'  *' if auc == best[0] else ''}", flush=True)
            save(ep + 1)
    model.load_state_dict(best[1])
    return model, evals_t, train_t, words, hist, best[0]


DEFAULT = dict(T=64, hidden=256, emb=256, depth=4, drop_p=0.1,
               lr=2e-3, wd=0.05, bs=128, epochs=100, ls=0.1,
               scale_s=30.0, margin=0.3, eval_every=5, seed=0, workers=4,
               speed=0.2, scale=0.1, rot=8.0, shift=0.05, drop=0.1, noise=0.01)


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULT.items():
        ap.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    ap.add_argument("--out", type=Path, default=Path("runs/base"))
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="resume file; written after every eval")
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULT}

    print(f"device={device()}  cfg={cfg}")
    t0 = time.time()
    model, evals, train_t, words, hist, best = train(cfg, ckpt=args.ckpt)
    res = evaluate(model, evals, train_t, words, device())

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "encoder.pt")
    json.dump(dict(cfg=cfg, best_val_auc=best, history=hist, results=res,
                   minutes=(time.time() - t0) / 60),
              open(args.out / "result.json", "w"), indent=2)

    print(f"\n{'split':10s} {'AUC':>7s} {'EER':>7s} {'rank-1':>7s}")
    for k, v in res.items():
        print(f"{k:10s} {v['auc']:7.4f} {v['eer']*100:6.2f}% {v['rank1']*100:6.2f}%")
    print(f"\n{(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
