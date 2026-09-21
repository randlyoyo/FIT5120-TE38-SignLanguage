#!/usr/bin/env python3
"""Render a clip's world landmarks as a skeleton.

The point is an upper bound. A generator trained on these clips can only ever
produce what they already contain, so if real signing rendered from
MediaPipe's world landmarks does not read as signing, no diffusion model over
them will either -- and the extractor, not the generator, is what needs
replacing. Front and side views are drawn together because the whole question
about these landmarks is whether the depth channel carries anything.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POSE_EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
              (0,11),(0,12),(7,8)]
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
              (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),
              (18,19),(19,20),(5,9),(9,13),(13,17)]
POSE_PTS = [0,7,8,11,12,13,14,15,16,23,24]


def panel(ax, pose, lh, rh, plane, box):
    # MediaPipe world: x right, y DOWN, z away from camera. Plot y negated so
    # the signer is upright; "front" is (x, -y), "side" is (z, -y).
    i = 0 if plane == "front" else 2
    P = lambda p: (p[..., i], -p[..., 1])
    for a, b in POSE_EDGES:
        ax.plot(*P(pose[[a, b]]), c="#2b2b2b", lw=2.2, solid_capstyle="round")
    ax.scatter(*P(pose[POSE_PTS]), s=14, c="#2b2b2b", zorder=3)
    for hand, col in ((lh, "#d1495b"), (rh, "#2274a5")):
        if np.isnan(hand[0, 0]):
            continue
        for a, b in HAND_EDGES:
            ax.plot(*P(hand[[a, b]]), c=col, lw=1.6, solid_capstyle="round")
        ax.scatter(*P(hand), s=7, c=col, zorder=4)
    (x0, x1), (y0, y1) = box
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal"); ax.set_axis_off()


def grid(path, word, out, n=8):
    d = np.load(path, allow_pickle=False)
    pose, lh, rh = d["pose_world"], d["left_hand_world"], d["right_hand_world"]
    meta = json.loads(str(d["meta"]))
    T = pose.shape[0]
    idx = np.linspace(0, T - 1, n).astype(int)

    # One shared frame for the whole clip, so motion is visible as motion
    # rather than cancelled out by per-frame rescaling.
    pts = np.concatenate([pose.reshape(-1, 3),
                          lh.reshape(-1, 3), rh.reshape(-1, 3)])
    pts = pts[~np.isnan(pts[:, 0])]
    cx, cy = np.median(pts[:, 0]), np.median(-pts[:, 1])
    r = 0.55
    box = ((cx - r, cx + r), (cy - r, cy + r))
    box_side = ((-r, r), (cy - r, cy + r))

    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.8))
    for j, t in enumerate(idx):
        panel(axes[0, j], pose[t], lh[t], rh[t], "front", box)
        panel(axes[1, j], pose[t], lh[t], rh[t], "side", box_side)
        axes[0, j].set_title(f"t={t}", fontsize=8)
    axes[0, 0].set_ylabel("front"); axes[1, 0].set_ylabel("side")
    fig.suptitle(f"{word}   {path.split('/')[-1]}   "
                 f"hand detection L{meta['left_hand_rate']:.2f} "
                 f"R{meta['right_hand_rate']:.2f}   {T} frames", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for cid, word in [("88500", "THANK YOU"), ("05670", "WHALE"),
                      ("79612", "WHAT"), ("76005", "MOTHER")]:
        grid(f"keypoints3d/Valid/{cid}_kf_rgb.npz", word,
             f"/tmp/skel_{word.replace(' ', '_')}.png")
