#!/usr/bin/env python3
"""Score every query in a split against every word's templates.

    python dtw_score.py --split Valid --out scores/Valid.npz

Writes the full (n_queries, n_words) distance matrix rather than a sampled set
of trials. It costs the same to compute -- the DTW work is dominated by the
templates, not by which columns we keep -- and it lets the evaluation stage
change the impostor count, pull hard negatives, or compute rank-1 accuracy
without re-running anything.

Templates always come from Train. Distance to a word is the MINIMUM over that
word's templates: verification asks whether the query matches *any* stored
example, and averaging lets one atypical template drag a good match down.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

import dtw_core as K
import dtw_features as F

LABELS = Path("MM-WLAuslan/labels")
SPLITS = ["Train", "Valid", "Test_ITW", "Test_STU",
          "Test_TED", "Test_SYN", "Test_MTV"]


def label_key(stem):
    """npz stem -> key used in the label json."""
    return stem[:-7] if stem.endswith("_kf_rgb") else stem


def load_labels(split):
    with open(LABELS / f"{split}.json") as fh:
        return json.load(fh)


def _build_one(path):
    try:
        return (path, F.build(path), None)
    except F.Unusable as e:
        return (path, None, str(e))
    except Exception as e:                                  # noqa: BLE001
        return (path, None, f"error: {e}")


def build_split(split, cache_dir="cache", workers=4):
    """Feature-build a whole split, cached. Templates are reused by every
    query split and by the cohort, so rebuilding 38k clips per run is waste."""
    cache = Path(cache_dir) / f"{split}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        offs, stems, buf = z["offsets"], list(z["stems"]), z["buf"]
        return [buf[offs[i]:offs[i + 1]] for i in range(len(stems))], stems

    from multiprocessing import Pool
    paths = sorted(Path(f"keypoints/{split}").glob("*.npz"))
    with Pool(workers) as pool:
        res = pool.map(_build_one, [str(p) for p in paths], chunksize=64)

    seqs, stems, dropped = [], [], {}
    for path, seq, err in res:
        if err:
            k = err.split(":")[0]
            dropped[k] = dropped.get(k, 0) + 1
            continue
        seqs.append(seq)
        stems.append(Path(path).stem)
    print(f"  {split}: {len(seqs)}/{len(paths)} usable, dropped {dropped}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    buf, offs = K.pack(seqs)
    np.savez_compressed(cache, buf=buf, offsets=offs, stems=np.array(stems))
    return seqs, stems


def group_by_word(seqs, stems, labels, n_template, seed=0):
    """word -> n_template sequences. Every word keeps the SAME count so the
    min-over-templates statistic stays comparable between words."""
    by = {}
    for s, stem in zip(seqs, stems):
        g = labels.get(label_key(stem))
        if g is not None:
            by.setdefault(g, []).append(s)
    rng = random.Random(seed)
    return {g: rng.sample(v, n_template)
            for g, v in by.items() if len(v) >= n_template}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-template", type=int, default=12)
    ap.add_argument("--cohort", type=int, default=300,
                    help="Train clips used to estimate per-word impostor stats")
    ap.add_argument("--n-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr_seqs, tr_stems = build_split("Train", args.cache, args.workers)
    tr_lab = load_labels("Train")
    templates = group_by_word(tr_seqs, tr_stems, tr_lab, args.n_template)
    words = sorted(templates)
    widx = {w: i for i, w in enumerate(words)}
    packed, offs = K.pack([t for w in words for t in templates[w]])
    print(f"  {len(words)} words x {args.n_template} templates")

    if args.split == "Train":
        q_seqs, q_stems, q_lab = tr_seqs, tr_stems, tr_lab
    else:
        q_seqs, q_stems = build_split(args.split, args.cache, args.workers)
        q_lab = load_labels(args.split)

    rng = random.Random(args.seed)

    # --- per-word impostor stats, for the optional z-normalisation ----------
    # Measured to be worth ~+0.003 AUC on this data, not the lever it is in
    # speaker verification -- the shoulder-width normalisation has already put
    # every word on a common scale. Kept because it is cheap and non-negative.
    cohort = rng.sample(list(zip(tr_seqs, tr_stems)),
                        min(args.cohort, len(tr_seqs)))
    print(f"estimating impostor stats from {len(cohort)} clips...")
    craw = np.empty((len(cohort), len(words)), np.float32)
    for i, (seq, _) in enumerate(cohort):
        craw[i] = K.score_against(seq, packed, offs, args.n_template)
    cw = [tr_lab.get(label_key(s)) for _, s in cohort]
    mu = np.empty(len(words), np.float32)
    sd = np.empty(len(words), np.float32)
    for j, w in enumerate(words):
        col = craw[np.array([c != w for c in cw]), j]
        mu[j], sd[j] = col.mean(), max(col.std(), 1e-6)

    # --- full distance matrix ----------------------------------------------
    qs = [(s, st) for s, st in zip(q_seqs, q_stems)
          if q_lab.get(label_key(st)) in widx]
    if args.n_queries:
        qs = rng.sample(qs, min(args.n_queries, len(qs)))
    print(f"scoring {len(qs)} queries x {len(words)} words...")

    D = np.empty((len(qs), len(words)), np.float32)
    truth = np.empty(len(qs), np.int32)
    for i, (seq, stem) in enumerate(qs):
        D[i] = K.score_against(seq, packed, offs, args.n_template)
        truth[i] = widx[q_lab[label_key(stem)]]
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(qs)}]", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, D=D, truth=truth, mu=mu, sd=sd,
        words=np.array(words), stems=np.array([st for _, st in qs]),
        split=args.split,
    )
    print(f"wrote {args.out}  D={D.shape}")


if __name__ == "__main__":
    main()
