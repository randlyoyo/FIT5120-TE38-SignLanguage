#!/usr/bin/env python3
"""Contact sheets for all 3,215 words, worst first.

    python contact_sheets.py --per-sheet 64

386 MB of animation cannot go in one page, and no one reviews 3,215 clips by
eye anyway. What a still sheet does catch is the gross failure -- a collapsed
pose, a hand that is not drawn, a limb in the wrong place -- and ordering the
sheets by quality score puts every one of those in the first few pages, so the
reviewer stops when the sheets go quiet rather than after 3,215 clips.

One frame per word, taken at peak hand elevation rather than the midpoint: a
sign's most informative instant is where the hands are highest, while the
midpoint of a clip is often still the rest pose.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import compare_video as CV
import retarget as RT

OUT = Path("renders/sheets")
RIG = json.load(open("renders/rig.json"))
REST = np.array(RIG["restOffsets"])


def peak_pose(word):
    """-> (positions at peak hand elevation, status there)"""
    d = json.load(open(f"renders/anim/{word.replace('/', '_')}.json"))
    q = np.array(d["rotations"])
    P, _ = RT.fk(q, REST, np.array(d["rootPositions"]))
    st = np.array(d["status"])
    ch = P[:, RT.IDX["chest"], 1]
    lift = np.maximum(ch - P[:, RT.IDX["wrist_L"], 1], ch - P[:, RT.IDX["wrist_R"], 1])
    t = int(np.nanargmax(lift))
    return P[t], st[t], t, len(q)


def sheet(args):
    page, rows, cols, items = args
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.55, rows * 1.95))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, it in zip(axes.ravel(), items):
        try:
            Pt, st, t, T = peak_pose(it["word"])
        except Exception:
            ax.text(.5, .5, "ERR\n" + it["word"][:14], ha="center", va="center",
                    fontsize=6, color="#c2453f", transform=ax.transAxes)
            continue
        for i, p in CV.BONES:
            if st[i] == 3 or st[p] == 3 or not np.isfinite(Pt[[i, p]]).all():
                continue
            ax.plot([Pt[i, 0], Pt[p, 0]], [-Pt[i, 1], -Pt[p, 1]],
                    c=CV.rig_colour(i), lw=1.5, solid_capstyle="round")
        hc = CV.head_circle(Pt)
        if hc:
            ax.add_patch(Circle(hc[0], hc[1], fill=False, ec="#2b2b2b", lw=1.2))
        # One window per panel, so a signer reaching high is not shrunk to fit.
        c = Pt[RT.IDX["chest"]]
        r = 0.62
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(-c[1] - r * 0.75, -c[1] + r * 1.25)
        ax.set_aspect("equal")
        col = ("#c2453f" if it["score"] < 0.72 or it["absent"] > 0.05
               else "#c8912f" if it["score"] < 0.80 else "#4a5260")
        ax.set_title(f"{it['word'][:17]}\n{it['score']:.2f}"
                     + (f"  {it['absent']*100:.0f}%abs" if it["absent"] > 0.005 else ""),
                     fontsize=6.2, color=col, pad=2)
    fig.suptitle(f"retargeted rig — peak frame — sheet {page['n']}/{page['of']}"
                 f"   score {items[0]['score']:.2f} to {items[-1]['score']:.2f}"
                 "   (red = score<0.72 or absent>5%)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = OUT / f"sheet_{page['n']:03d}.png"
    fig.savefig(p, dpi=104)
    plt.close(fig)
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sheet", type=int, default=64)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    words = json.load(open("renders/manifest.json"))["words"]
    words.sort(key=lambda w: (w["score"], -w["absent"]))     # worst first
    rows = -(-a.per_sheet // a.cols)
    chunks = [words[i:i + a.per_sheet] for i in range(0, len(words), a.per_sheet)]
    jobs = [({"n": i + 1, "of": len(chunks)}, rows, a.cols, c)
            for i, c in enumerate(chunks)]
    print(f"{len(words)} words -> {len(chunks)} sheets of {a.per_sheet}", flush=True)
    with Pool(a.workers) as pool:
        for i, p in enumerate(pool.imap_unordered(sheet, jobs)):
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(chunks)}", flush=True)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
