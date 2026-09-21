#!/usr/bin/env python3
"""Animate retargeted skeletons -- no character model required.

    python animate.py /tmp/rt_return.npz /tmp/rt_return_s.npz --out /tmp/x.gif

A skeleton is drawable straight from joint positions, which is the whole
reason to validate the motion at this stage rather than after rigging: a
character model would add its own failure modes on top of the ones being
judged here.

Still frames cannot settle whether the smoothing is right. Jitter is a
property of consecutive frames, and a filter that is too aggressive looks
identical to a good one in a contact sheet and only reveals itself as
mushiness in motion. Hence the two columns: same clip, filter off and on.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
    return HANDCOL[n[-1]] if RT.SKEL[i][2][0] == "hand" else "#2b2b2b"


ABSENT, INTERPOLATED, HELD = 3, 1, 2


class Rig:
    """One pane: pre-built line artists updated per frame.

    A bone with an ABSENT endpoint is cleared rather than drawn, so a hand that
    was never detected simply is not there. Guessed joints are faded.
    """

    def __init__(self, ax, P, hand=None, lw=1.8, status=None):
        self.P, self.hand, self.status = P, hand, status
        self.lines = [ax.plot([], [], c=colour(i), lw=lw,
                              solid_capstyle="round")[0] for i, _ in BONES]
        ax.set_aspect("equal"); ax.set_axis_off()
        self.ax = ax
        if hand:
            self.w = RT.IDX[f"wrist_{hand}"]
            self.r = 0.11
        else:
            pts = P.reshape(-1, 3); pts = pts[np.isfinite(pts[:, 0])]
            self.cx, self.cy = np.median(pts[:, 0]), np.median(-pts[:, 1])
            self.r = 0.55

    def update(self, t):
        P = self.P[t]
        st = None if self.status is None else self.status[t]
        for ln, (i, p) in zip(self.lines, BONES):
            drawable = np.isfinite(P[i]).all() and np.isfinite(P[p]).all()
            if st is not None and (st[i] == ABSENT or st[p] == ABSENT):
                drawable = False
            if drawable:
                ln.set_data([P[i, 0], P[p, 0]], [-P[i, 1], -P[p, 1]])
                guess = st is not None and (st[i] in (INTERPOLATED, HELD)
                                            or st[p] in (INTERPOLATED, HELD))
                ln.set_alpha(0.30 if guess else 1.0)
            else:
                ln.set_data([], [])
        if self.hand:
            cx, cy = P[self.w, 0], -P[self.w, 1]
        else:
            cx, cy = self.cx, self.cy
        self.ax.set_xlim(cx - self.r, cx + self.r)
        self.ax.set_ylim(cy - self.r, cy + self.r)
        return self.lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="one or two .npz from retarget.py")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="/tmp/rig.gif")
    ap.add_argument("--hand", default="R")
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args()

    runs = [np.load(p) for p in a.npz]
    labels = a.labels or (["raw", "smoothed"][:len(runs)] if len(runs) > 1
                          else ["retargeted"])
    T = min(r["P_rig"].shape[0] for r in runs)

    fig, axes = plt.subplots(2, len(runs), figsize=(3.0 * len(runs), 5.6),
                             squeeze=False)
    panes = []
    for c, r in enumerate(runs):
        st = r["status"] if "status" in r else None
        panes.append(Rig(axes[0][c], r["P_rig"], status=st))
        panes.append(Rig(axes[1][c], r["P_rig"], hand=a.hand, lw=2.6, status=st))
        axes[0][c].set_title(labels[c], fontsize=11)
    fig.tight_layout()

    def frame(t):
        arts = []
        for p in panes:
            arts += p.update(t)
        return arts

    ani = FuncAnimation(fig, frame, frames=T, interval=1000 / a.fps, blit=False)
    ani.save(a.out, writer=PillowWriter(fps=a.fps), dpi=80)
    print(f"wrote {a.out}  ({T} frames)")


if __name__ == "__main__":
    main()
