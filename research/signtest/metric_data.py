#!/usr/bin/env python3
"""Batching and augmentation for the metric-learning encoder.

Reads the same cached feature sequences the DTW baseline uses, so the two are
scored on identical inputs and the comparison stays honest.

Augmentation matters more than usual here: 3215 classes with 12 examples each
is small enough that a 1.3M-parameter encoder will simply memorise the training
clips without it.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import dtw_features as F

# The 2D and 3D experiments differ only in which cache they read and how many
# coordinates a point has, so both are environment switches rather than a
# forked copy of this file. Defaults reproduce the 2D pipeline exactly.
CACHE = Path(os.environ.get("SIGN_CACHE", "cache"))
LABELS = Path("MM-WLAuslan/labels")
NDIM = int(os.environ.get("SIGN_NDIM", "2"))          # 2 = image plane, 3 = world
EVAL_SPLITS = tuple(os.environ.get(
    "SIGN_SPLITS", "Valid,Test_STU,Test_ITW,Test_TED,Test_SYN").split(","))

# Layout of the 98-dim frame vector, needed to augment coordinates without
# touching the presence flags.
N_ARM, N_HAND = F.N_ARM, F.N_HAND
N_PT = N_ARM + 2 * N_HAND          # 48 points
N_XY = N_PT * NDIM                 # coordinate dims; the last 2 are flags


def label_key(stem):
    return stem[:-7] if stem.endswith("_kf_rgb") else stem


def load_split(split):
    """-> (list of (T,98) arrays, list of stems)"""
    z = np.load(CACHE / f"{split}.npz", allow_pickle=False)
    offs, stems, buf = z["offsets"], list(z["stems"]), z["buf"]
    return [buf[offs[i]:offs[i + 1]] for i in range(len(stems))], stems


def load_labels(split):
    with open(LABELS / f"{split}.json") as fh:
        return json.load(fh)


def resample(x, T):
    """Linear index resample to exactly T frames.

    Fixed length is what makes a single embedding possible at all, and it is
    the real cost of leaving DTW behind: the encoder has to learn the temporal
    alignment that DTW got for free. Resampling rather than cropping keeps the
    whole sign in view -- a crop would cut the end off long signs, and the end
    of a sign is often what distinguishes it.
    """
    n = x.shape[0]
    if n == T:
        return x
    idx = np.linspace(0, n - 1, T)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (idx - lo).astype(np.float32)[:, None]
    return x[lo] * (1 - w) + x[hi] * w


class ClipSet(Dataset):
    def __init__(self, seqs, labels, T=64, train=False, aug=None):
        self.seqs, self.labels, self.T, self.train = seqs, labels, T, train
        self.aug = aug or {}

    def __len__(self):
        return len(self.seqs)

    def _augment(self, x):
        a = self.aug
        rng = np.random

        # --- temporal: speed jitter, by resampling to a different length
        # before the fixed-length resample. Signing tempo genuinely varies
        # between people, so this is realistic rather than synthetic noise.
        if a.get("speed", 0) > 0:
            f = 1.0 + rng.uniform(-a["speed"], a["speed"])
            x = resample(x, max(8, int(round(x.shape[0] * f))))

        x = resample(x, self.T).copy()
        xy = x[:, :N_XY].reshape(self.T, N_PT, NDIM)

        # --- spatial: scale / rotate / shift, applied to the whole body so
        # the pose stays internally consistent.
        if a.get("scale", 0) > 0:
            xy *= 1.0 + rng.uniform(-a["scale"], a["scale"])
        if a.get("rot", 0) > 0:
            t = np.deg2rad(rng.uniform(-a["rot"], a["rot"]))
            c, s = np.cos(t), np.sin(t)
            if NDIM == 2:
                R = np.array([[c, -s], [s, c]], np.float32)
            else:
                # About the body's vertical axis (index 1 after canonicalisation):
                # an in-plane roll would be a camera artefact the 3D frame has
                # already removed, whereas azimuth is the variation MTV shows.
                R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], np.float32)
            xy = xy @ R
        if a.get("shift", 0) > 0:
            xy += rng.uniform(-a["shift"], a["shift"], size=(1, 1, NDIM)).astype(np.float32)

        # --- frame dropout: blank whole frames, mimicking the detector losing
        # the hands. Real clips average 0.79 hand detection, so the encoder has
        # to stay usable with gaps rather than assume clean input.
        if a.get("drop", 0) > 0:
            m = rng.random(self.T) < a["drop"]
            xy[m] = 0.0

        if a.get("noise", 0) > 0:
            xy += rng.normal(0, a["noise"], xy.shape).astype(np.float32)

        x[:, :N_XY] = xy.reshape(self.T, N_XY)
        return x

    def __getitem__(self, i):
        x = self.seqs[i]
        x = self._augment(x) if self.train else resample(x, self.T)
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), self.labels[i]


def build_sets(T=64, aug=None, splits=EVAL_SPLITS):
    """Train set plus every evaluation split, sharing one word vocabulary.

    Test_MTV is deliberately absent: it is a multi-view capture and the earlier
    diagnosis (shoulder foreshortening 0.73-3.10 against 1.68-1.82 everywhere
    else) showed 2D keypoints cannot bridge that. Keeping it in the loop would
    only add noise to model selection.
    """
    tr_seqs, tr_stems = load_split("Train")
    tr_lab = load_labels("Train")
    words = sorted({tr_lab[label_key(s)] for s in tr_stems
                    if label_key(s) in tr_lab})
    widx = {w: i for i, w in enumerate(words)}

    keep = [(s, widx[tr_lab[label_key(st)]])
            for s, st in zip(tr_seqs, tr_stems) if label_key(st) in tr_lab]
    train = ClipSet([a for a, _ in keep], [b for _, b in keep], T, True, aug)

    evals = {}
    for sp in splits:
        p = CACHE / f"{sp}.npz"
        if not p.exists():
            continue
        seqs, stems = load_split(sp)
        lab = load_labels(sp)
        pair = [(s, widx[lab[label_key(st)]]) for s, st in zip(seqs, stems)
                if lab.get(label_key(st)) in widx]
        evals[sp] = ClipSet([a for a, _ in pair], [b for _, b in pair], T)
    return train, evals, words


# ---------------------------------------------------------------------------
# Fixed-length tensors + GPU batch augmentation.
#
# The per-sample numpy path above costs ~74 s/epoch against ~8 s of GPU compute
# -- the accelerator idles while one Python process resamples 38k variable-length
# clips. Resampling once up front and augmenting whole batches on device removes
# that gap, which is what makes an unattended sweep affordable.
# ---------------------------------------------------------------------------

def load_fixed(split, T, cache_dir=CACHE):
    """(N,T,D) float32 + stems, resampled once and cached per T."""
    p = Path(cache_dir) / f"fixed_{split}_T{T}.npy"
    q = Path(cache_dir) / f"fixed_{split}_T{T}.stems.npy"
    if p.exists() and q.exists():
        return np.load(p), list(np.load(q))
    seqs, stems = load_split(split)
    X = np.stack([resample(s, T) for s in seqs]).astype(np.float32)
    np.save(p, X)
    np.save(q, np.array(stems))
    return X, stems


def build_fixed(T, splits=EVAL_SPLITS):
    """Train tensor + eval tensors + shared vocabulary. Test_MTV is excluded:
    it is a multi-view capture that 2D keypoints cannot bridge, so letting it
    into model selection would only add noise."""
    Xtr, tr_stems = load_fixed("Train", T)
    tr_lab = load_labels("Train")
    words = sorted({tr_lab[label_key(s)] for s in tr_stems
                    if label_key(s) in tr_lab})
    widx = {w: i for i, w in enumerate(words)}
    keep = [i for i, s in enumerate(tr_stems) if label_key(s) in tr_lab]
    Xtr = Xtr[keep]
    ytr = np.array([widx[tr_lab[label_key(tr_stems[i])]] for i in keep], np.int64)

    evals = {}
    for sp in splits:
        if not (Path(CACHE) / f"{sp}.npz").exists():
            continue
        X, stems = load_fixed(sp, T)
        lab = load_labels(sp)
        k = [i for i, s in enumerate(stems) if lab.get(label_key(s)) in widx]
        evals[sp] = (X[k],
                     np.array([widx[lab[label_key(stems[i])]] for i in k], np.int64))
    return (Xtr, ytr), evals, words


def augment_gpu(x, a):
    """Batch augmentation on (B,T,D) in place-ish, on whatever device x is on.

    Same transformations as the per-sample path, expressed as batched tensor ops
    so the cost lands on the GPU instead of a Python loop.
    """
    import torch
    B, T, _ = x.shape
    dev = x.device
    xy = x[:, :, :N_XY].reshape(B, T, N_PT, NDIM)

    if a.get("speed", 0) > 0:
        # Temporal jitter as a per-sample resample of the time axis: crop a
        # window of random width and stretch it back to T.
        f = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * a["speed"]
        base = torch.linspace(0, 1, T, device=dev)[None, :]
        pos = ((base - 0.5) * f[:, None] + 0.5).clamp(0, 1) * (T - 1)
        lo = pos.floor().long()
        hi = (lo + 1).clamp(max=T - 1)
        w = (pos - lo.float())[:, :, None, None]
        bi = torch.arange(B, device=dev)[:, None]
        xy = xy[bi, lo] * (1 - w) + xy[bi, hi] * w

    if a.get("scale", 0) > 0:
        s = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * a["scale"]
        xy = xy * s
    if a.get("rot", 0) > 0:
        t = (torch.rand(B, device=dev) * 2 - 1) * (a["rot"] * math.pi / 180)
        c, s = torch.cos(t), torch.sin(t)
        if NDIM == 2:
            R = torch.stack([torch.stack([c, -s], -1),
                             torch.stack([s, c], -1)], -2)          # (B,2,2)
        else:
            z, o = torch.zeros_like(c), torch.ones_like(c)
            # Yaw about the canonical vertical axis -- see the numpy path.
            R = torch.stack([torch.stack([c, z, -s], -1),
                             torch.stack([z, o, z], -1),
                             torch.stack([s, z, c], -1)], -2)       # (B,3,3)
        xy = torch.einsum("btpk,bkj->btpj", xy, R)
    if a.get("shift", 0) > 0:
        xy = xy + (torch.rand(B, 1, 1, NDIM, device=dev) * 2 - 1) * a["shift"]
    if a.get("drop", 0) > 0:
        m = (torch.rand(B, T, 1, 1, device=dev) < a["drop"])
        xy = xy.masked_fill(m, 0.0)
    if a.get("noise", 0) > 0:
        xy = xy + torch.randn_like(xy) * a["noise"]

    return torch.cat([xy.reshape(B, T, N_XY), x[:, :, N_XY:]], dim=2)
