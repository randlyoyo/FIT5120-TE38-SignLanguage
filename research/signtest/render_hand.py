#!/usr/bin/env python3
"""Zoom on one hand across a clip.

Body motion reading correctly is not enough: handshape is a phonological
parameter, so if the 21 hand landmarks do not resolve into fingers the data
cannot teach a sign, however good the trajectory looks.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),
         (11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),
         (5,9),(9,13),(13,17)]
FINGER = {"thumb":(1,5),"index":(5,9),"middle":(9,13),"ring":(13,17),"pinky":(17,21)}
COL = {"thumb":"#e07a5f","index":"#3d405b","middle":"#81b29a",
       "ring":"#f2cc8f","pinky":"#9d4edd"}


def grid(path, word, side, out, n=8):
    d = np.load(path, allow_pickle=False)
    h = d[f"{side}_hand_world"]
    meta = json.loads(str(d["meta"]))
    ok = np.flatnonzero(~np.isnan(h[:, 0, 0]))
    if len(ok) == 0:
        print(f"  {word}: no {side} hand at all"); return
    idx = ok[np.linspace(0, len(ok) - 1, min(n, len(ok))).astype(int)]

    # One window for the whole clip, sized from the hand's own extent, so a
    # closing fist reads as closing rather than being rescaled back open.
    rel = h[idx] - h[idx][:, :1]
    R = float(np.nanmax(np.abs(rel))) * 1.15
    fig, axes = plt.subplots(2, len(idx), figsize=(1.9 * len(idx), 4.4))
    axes = np.atleast_2d(axes)
    for j, t in enumerate(idx):
        p = h[t] - h[t, 0]                     # wrist at origin
        for row, (a, b, lab) in enumerate([(0, 1, "palm-on (x,-y)"),
                                           (2, 1, "side (z,-y)")]):
            ax = axes[row, j]
            for u, v in EDGES:
                ax.plot([p[u, a], p[v, a]], [-p[u, b], -p[v, b]],
                        c="#bbb", lw=1, zorder=1)
            for name, (lo, hi) in FINGER.items():
                ax.plot(p[lo:hi, a], -p[lo:hi, b], c=COL[name], lw=2.4,
                        solid_capstyle="round", zorder=2)
            ax.scatter(p[:, a], -p[:, b], s=9, c="#333", zorder=3)
            ax.set_xlim(-R, R); ax.set_ylim(-R, R)
            ax.set_aspect("equal"); ax.set_axis_off()
            if j == 0:
                ax.text(-R, -R, lab, fontsize=8, color="#666")
        axes[0, j].set_title(f"t={t}", fontsize=8)
    rate = meta[f"{side}_hand_rate"]
    fig.suptitle(f"{word}   {side} hand   detection {rate:.2f}   "
                 f"window ±{R:.3f} m", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for cid, word, side in [("88500", "THANK YOU", "right"),
                            ("79612", "WHAT", "right"),
                            ("76005", "MOTHER", "right")]:
        grid(f"keypoints3d/Valid/{cid}_kf_rgb.npz", word, side,
             f"/tmp/hand_{word.replace(' ','_')}.png")
