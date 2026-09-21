#!/usr/bin/env python3
"""Draw the retargeted skeleton against the landmarks it came from.

Position error is the wrong yardstick here: retargeting deliberately imposes
fixed bone lengths on a measurement whose lengths wobble ~17% frame to frame,
so it CANNOT match positions and should not try. What matters is whether the
result reads as the same sign and holds still when the signer holds still,
which is a question for the eye.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import retarget as RT

# Drawing edges, which are not the same as the rig's bones. The skeleton has
# one joint at the shoulder midpoint and one at the hip midpoint, so drawing
# only parent-child links renders the torso as a bare stem with a V on top --
# anatomically correct and completely unreadable. These two extra segments
# close the shoulder line and give the pelvis width, for the picture only.
BONES = ([(i, RT.PARENT[i]) for i in range(len(RT.SKEL)) if RT.PARENT[i] >= 0]
         + [(RT.IDX["shoulder_L"], RT.IDX["shoulder_R"])])

# The rig's "neck" joint sits at the nose, so without a drawn head the spine
# and the neck read as one pole running through the figure. The circle is
# cosmetic -- the corpus has no 3D head landmarks to size it from (MediaPipe
# emits no world landmarks for the face), so it is scaled off shoulder width.
HEAD_R = 0.30       # of shoulder width


def view_box(P, pad=1.18):
    """Window covering the whole clip plus the drawn head.

    Centring on the median and using a fixed radius clipped the head off the
    top: the figure's extent is not symmetric about its own median, and the
    head circle is drawn outside every joint.
    """
    pts = P.reshape(-1, 3)
    pts = pts[np.isfinite(pts[:, 0])]
    x, y = pts[:, 0], -pts[:, 1]
    sw = np.nanmedian(np.linalg.norm(
        P[:, RT.IDX["shoulder_L"]] - P[:, RT.IDX["shoulder_R"]], axis=-1))
    m = HEAD_R * sw * 1.7
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    r = max(x.max() - x.min(), y.max() - y.min() + m) / 2 * pad
    return (cx - r, cx + r), (cy - r + m / 2, cy + r + m / 2)


def head_circle(P):
    a, b = P[RT.IDX["shoulder_L"]], P[RT.IDX["shoulder_R"]]
    n = P[RT.IDX["neck"]]
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(n).all()):
        return None
    r = HEAD_R * np.linalg.norm(a - b)
    ch = (a + b) / 2.0
    up = (n - ch)
    up = up / max(np.linalg.norm(up), 1e-9)
    c = n + up * r * 0.55
    return (c[0], -c[1]), r
HANDCOL = {"L": "#d1495b", "R": "#2274a5"}


def colour(i):
    n = RT.NAMES[i]
    if RT.SKEL[i][2][0] == "hand":
        return HANDCOL[n[-1]]
    return "#2b2b2b"


ABSENT, INTERPOLATED, HELD = 3, 1, 2


def panel(ax, P, box, lw=1.8, st=None):
    """A bone whose endpoints are ABSENT is not drawn at all.

    Showing a fabricated hand is worse than showing none: the viewer of a
    teaching app cannot tell them apart, and one of them is a wrong sign.
    Interpolated and held joints are drawn faded, so they read as "we are not
    sure" rather than as instruction.
    """
    for i, p in BONES:
        if not (np.isfinite(P[i]).all() and np.isfinite(P[p]).all()):
            continue
        if st is not None and (st[i] == ABSENT or st[p] == ABSENT):
            continue
        guess = st is not None and (st[i] in (INTERPOLATED, HELD)
                                    or st[p] in (INTERPOLATED, HELD))
        ax.plot([P[i, 0], P[p, 0]], [-P[i, 1], -P[p, 1]],
                c=colour(i), lw=lw, solid_capstyle="round",
                alpha=0.30 if guess else 1.0,
                ls=(0, (2, 2)) if guess else "-")
    ok = np.isfinite(P).all(axis=-1)
    if st is not None:
        ok &= st != ABSENT
    ax.scatter(P[ok, 0], -P[ok, 1], s=6, c=[colour(i) for i in np.flatnonzero(ok)],
               zorder=3)
    (x0, x1), (y0, y1) = box
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal"); ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="the .npz written by retarget.py")
    ap.add_argument("--out", default="/tmp/retarget_compare.png")
    ap.add_argument("--title", default="")
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--hand", choices=["L", "R"], default=None,
                    help="zoom on one hand instead of the whole body")
    a = ap.parse_args()

    d = np.load(a.npz)
    src, rig = d["P_src"], d["P_rig"]
    status = d["status"] if "status" in d else None
    T = src.shape[0]
    idx = np.linspace(0, T - 1, a.n).astype(int)

    if a.hand:
        # Follow the wrist: a hand held still in a fixed world window is a few
        # pixels across, and finger detail is the whole point of the zoom.
        w = RT.IDX[f"wrist_{a.hand}"]
        sel = [i for i in range(len(RT.SKEL))
               if RT.NAMES[i].endswith(f"_{a.hand}") and RT.SKEL[i][2][0] == "hand"]
        r = 0.11
        boxes = [((src[t, w, 0] - r, src[t, w, 0] + r),
                  (-src[t, w, 1] - r, -src[t, w, 1] + r)) for t in range(T)]
    else:
        pts = np.concatenate([src.reshape(-1, 3), rig.reshape(-1, 3)])
        pts = pts[np.isfinite(pts[:, 0])]
        cx, cy = np.median(pts[:, 0]), np.median(-pts[:, 1])
        r = 0.55
        boxes = [((cx - r, cx + r), (cy - r, cy + r))] * T

    fig, axes = plt.subplots(2, len(idx), figsize=(1.8 * len(idx), 4.0))
    for j, t in enumerate(idx):
        panel(axes[0, j], src[t], boxes[t], lw=2.4 if a.hand else 1.8)
        panel(axes[1, j], rig[t], boxes[t], lw=2.4 if a.hand else 1.8,
              st=None if status is None else status[t])
        tag = ""
        if status is not None:
            if (status[t] == ABSENT).any():
                tag = "  absent"
            elif (status[t] != 0).any():
                tag = "  guessed"
        axes[0, j].set_title(f"t={t}{tag}", fontsize=8)
    axes[0, 0].text(boxes[idx[0]][0][0], boxes[idx[0]][1][0], "landmarks",
                    fontsize=9, color="#666")
    axes[1, 0].text(boxes[idx[0]][0][0], boxes[idx[0]][1][0], "retargeted rig",
                    fontsize=9, color="#666")
    fig.suptitle(a.title or a.npz, fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
