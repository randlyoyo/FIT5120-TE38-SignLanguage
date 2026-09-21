#!/usr/bin/env python3
"""Turn distance matrices into the verification numbers that decide next steps.

    python dtw_eval.py --scores scores/*.npz --out checks/dtw

Reports, per split: AUC, EER, and the FAR/FRR you actually get when the
threshold is fixed on Valid and applied unchanged -- which is the only honest
operating point, since picking a threshold on the test set is self-scoring.

ROC/AUC/EER rather than precision/recall on purpose: TPR and FPR are each
normalised within their own class, so the numbers do not move when the
impostor-per-query ratio changes. A PR curve would.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

N_NEG = 50           # impostor words sampled per query
SEED = 0


def trials(D, truth, n_neg=N_NEG, mu=None, sd=None, seed=SEED):
    """(scores, labels) for one genuine + n_neg impostor trials per query.

    Impostors are drawn uniformly from the other words. Uniform on purpose:
    it is an unbiased estimate of the real impostor distribution. Deliberately
    picking look-alike signs makes a more pessimistic number that belongs in a
    separate analysis, not in the headline metric.
    """
    rng = np.random.default_rng(seed)
    n_q, n_w = D.shape
    S = D if mu is None else (D - mu) / sd
    neg = rng.integers(0, n_w, size=(n_q, n_neg + 1))
    ok = neg != truth[:, None]
    s, y = [], []
    for i in range(n_q):
        cols = neg[i][ok[i]][:n_neg]
        s.append(S[i, truth[i]])
        y.append(1)
        s.extend(S[i, cols])
        y.extend([0] * len(cols))
    return np.array(s), np.array(y, np.int8)


def metrics(s, y):
    """AUC + EER. Distance is a dissimilarity, so similarity is -distance."""
    auc = roc_auc_score(y, -s)
    fpr, tpr, thr = roc_curve(y, -s)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = (fpr[i] + (1 - tpr[i])) / 2
    return auc, eer, -thr[i], (fpr, tpr)


def at_threshold(s, y, tau):
    """FAR/FRR at a threshold carried over from another split."""
    acc = s < tau
    far = float(acc[y == 0].mean())
    frr = float((~acc[y == 1]).mean())
    return far, frr


def per_word_eer(D, truth, words, mu=None, sd=None, min_q=2):
    """EER computed word by word. The mean hides that some words never work."""
    S = D if mu is None else (D - mu) / sd
    out = {}
    for j in range(len(words)):
        q = np.flatnonzero(truth == j)
        if len(q) < min_q:
            continue
        pos = S[q, j]
        neg = np.delete(S[q], j, axis=1).ravel()
        s = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        fpr, tpr, _ = roc_curve(y, -s)
        i = np.nanargmin(np.abs(fpr - (1 - tpr)))
        out[words[j]] = (fpr[i] + (1 - tpr[i])) / 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("checks/dtw"))
    ap.add_argument("--n-neg", type=int, default=N_NEG)
    ap.add_argument("--znorm", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted({p for g in args.scores for p in glob.glob(g)})
    order = ["Valid", "Test_STU", "Test_ITW", "Test_TED", "Test_SYN", "Test_MTV"]
    data = {}
    for p in paths:
        z = np.load(p, allow_pickle=False)
        data[str(z["split"])] = z
    splits = [s for s in order if s in data] + \
             [s for s in data if s not in order]

    # Threshold is fixed on Valid, then applied everywhere else untouched.
    tau = None
    rows, roc_data = [], {}
    for s in splits:
        z = data[s]
        mu, sd = (z["mu"], z["sd"]) if args.znorm else (None, None)
        sc, y = trials(z["D"], z["truth"], args.n_neg, mu, sd)
        auc, eer, t, (fpr, tpr) = metrics(sc, y)
        if s == "Valid":
            tau = t
        far, frr = at_threshold(sc, y, tau) if tau is not None else (np.nan,) * 2
        # rank-1 identification: a free sanity check on the distance itself
        r1 = float((z["D"].argmin(axis=1) == z["truth"]).mean())
        rows.append(dict(split=s, n_query=int(z["D"].shape[0]),
                         auc=auc, eer=eer, far=far, frr=frr, rank1=r1))
        roc_data[s] = (fpr, tpr)
        print(f"{s:10s} n={z['D'].shape[0]:6d}  AUC={auc:.4f}  EER={eer*100:5.2f}%"
              f"  @Valid-tau FAR={far*100:5.2f}% FRR={frr*100:5.2f}%"
              f"  rank-1={r1*100:5.2f}%")

    with open(args.out / "summary.json", "w") as fh:
        json.dump({"tau_from_valid": float(tau), "n_neg": args.n_neg,
                   "znorm": args.znorm, "rows": rows}, fh, indent=2)

    _plots(data, roc_data, rows, args, tau)
    print(f"\nwrote {args.out}/")


def _plots(data, roc_data, rows, args, tau):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # ROC
    for s, (fpr, tpr) in roc_data.items():
        r = next(r for r in rows if r["split"] == s)
        ax[0, 0].plot(fpr, tpr, lw=1.6, label=f"{s} (AUC {r['auc']:.3f})")
    ax[0, 0].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[0, 0].set(xlabel="FAR", ylabel="1 - FRR", title="ROC")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)

    # DET -- normal deviate axes, which is where verification systems are read
    for s, (fpr, tpr) in roc_data.items():
        m = (fpr > 1e-4) & (fpr < 1) & (tpr > 0) & (tpr < 1)
        ax[0, 1].plot(norm.ppf(fpr[m]), norm.ppf(1 - tpr[m]), lw=1.6, label=s)
    tick = [.01, .05, .1, .2, .4, .6]
    ax[0, 1].set_xticks(norm.ppf(tick)); ax[0, 1].set_xticklabels([f"{t:.0%}" for t in tick])
    ax[0, 1].set_yticks(norm.ppf(tick)); ax[0, 1].set_yticklabels([f"{t:.0%}" for t in tick])
    ax[0, 1].set(xlabel="FAR", ylabel="FRR", title="DET")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)

    # score distributions, in-domain vs the hardest split
    for k, s in enumerate([x for x in ["Valid", "Test_MTV"] if x in data]):
        z = data[s]
        mu, sd = (z["mu"], z["sd"]) if args.znorm else (None, None)
        sc, y = trials(z["D"], z["truth"], args.n_neg, mu, sd)
        a = ax[1, 0]
        a.hist(sc[y == 1], bins=60, alpha=.5, density=True,
               label=f"{s} genuine", color=f"C{k*2}")
        a.hist(sc[y == 0], bins=60, alpha=.35, density=True,
               label=f"{s} impostor", color=f"C{k*2+1}")
    ax[1, 0].axvline(tau, color="k", ls="--", lw=1, label="tau (from Valid)")
    ax[1, 0].set(xlabel="score (DTW distance)", ylabel="density",
                 title="Genuine vs impostor")
    ax[1, 0].legend(fontsize=8)

    # per-word EER: the mean hides words that never work at all
    for s in [x for x in ["Valid", "Test_MTV"] if x in data]:
        z = data[s]
        mu, sd = (z["mu"], z["sd"]) if args.znorm else (None, None)
        pw = per_word_eer(z["D"], z["truth"], list(z["words"]), mu, sd)
        v = np.array(list(pw.values()))
        ax[1, 1].hist(v * 100, bins=40, alpha=.55,
                      label=f"{s} (median {np.median(v)*100:.1f}%)")
        json.dump({k: float(x) for k, x in pw.items()},
                  open(args.out / f"per_word_eer_{s}.json", "w"), indent=1)
    ax[1, 1].set(xlabel="per-word EER (%)", ylabel="words",
                 title="Per-word EER distribution")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out / "dtw_baseline.png", dpi=140)


if __name__ == "__main__":
    main()
