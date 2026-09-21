#!/usr/bin/env python3
"""Original video, detected landmarks, and retargeted rig, side by side.

    python compare_video.py 93128 --out renders/compare_93128.gif

The skeleton animations on their own answer the wrong question. They show
whether the rig is self-consistent, not whether it matches what the signer
actually did -- and every stage before it (detection, world-landmark lifting,
frame solving, smoothing) can fail in ways that still look plausible in
isolation. Putting the source frames next to the result is the only check that
covers the whole chain at once.

Frame indices are 1:1: the extractor stores one row per decoded frame and
retarget.py never resamples, so panel n of each column is the same instant.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

import retarget as RT
import slr_common as C

P11 = {v: i for i, v in enumerate(C.POSE_SUBSET)}
ARM = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
              (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),
              (18,19),(19,20),(5,9),(9,13),(13,17)]
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


FACE_CH = ["browRaise", "browAsym", "mouthOpen", "mouthWide",
           "headYaw", "headPitch", "headRoll"]


# ARKit blendshape names this renderer reads. FaceLandmarker emits all 52; a
# stick figure can only show a few, and these are the ones that carry meaning
# in Auslan -- brows mark question type, the jaw and lips carry mouthing, and
# eye aperture is the one channel the 18-point face subset could not express
# at all, so blinks had to be invented before this existed.
BS_READ = ("browInnerUp", "browDownLeft", "browDownRight",
           "browOuterUpLeft", "browOuterUpRight",
           "eyeBlinkLeft", "eyeBlinkRight",
           "jawOpen", "mouthPucker", "mouthSmileLeft", "mouthSmileRight",
           "mouthFrownLeft", "mouthFrownRight")


def draw_face_bs(ax, P, bs, names, rpy, artists=None):
    """Draw the head from measured blendshapes rather than from the 7-channel
    approximation.

    The approximation inferred brow height from six landmarks and mouth
    opening from twelve; these weights come from a model trained to produce
    them, include the eyes, and are already in the range a rig expects.
    """
    hc = head_circle(P)
    if hc is None:
        return artists
    (cx, cy), r = hc
    g = {n: float(bs[names.index(n)]) if n in names else 0.0 for n in BS_READ}
    roll = np.radians(rpy[2]) if rpy is not None else 0.0
    yaw = np.radians(rpy[1]) if rpy is not None else 0.0
    pitch = np.radians(rpy[0]) if rpy is not None else 0.0
    co, si = np.cos(roll), np.sin(roll)

    def place(dx, dy):
        dx = dx + np.sin(yaw) * 0.5
        dy = dy - np.sin(pitch) * 0.5
        return cx + (dx * co - dy * si) * r, cy + (dx * si + dy * co) * r

    up = g["browInnerUp"]
    outL, outR = g["browOuterUpLeft"], g["browOuterUpRight"]
    dnL, dnR = g["browDownLeft"], g["browDownRight"]
    # A brow is drawn as a segment whose inner and outer ends move separately:
    # in Auslan the shape matters, not just the height -- inner-up reads as a
    # yes/no question, both-down as a wh-question.
    segs = []
    for side, out_, dn in ((-1, outL, dnL), (1, outR, dnR)):
        yin = 0.26 + up * 0.20 - dn * 0.20
        yout = 0.26 + out_ * 0.20 - dn * 0.20
        segs.append(([place(side * 0.18, yin)[0], place(side * 0.52, yout)[0]],
                     [place(side * 0.18, yin)[1], place(side * 0.52, yout)[1]]))

    eyes = []
    for side, blink in ((-1, g["eyeBlinkLeft"]), (1, g["eyeBlinkRight"])):
        h = (1.0 - blink) * 0.09 + 0.01
        t = np.linspace(0, 2 * np.pi, 16)
        ex, ey = [], []
        for aa in t:
            x, y = place(side * 0.34 + np.cos(aa) * 0.13, 0.06 + np.sin(aa) * h)
            ex.append(x); ey.append(y)
        eyes.append((ex, ey))

    open_ = g["jawOpen"] * 0.20 + 0.015
    wide = 0.30 - g["mouthPucker"] * 0.14
    smile = (g["mouthSmileLeft"] + g["mouthSmileRight"] -
             g["mouthFrownLeft"] - g["mouthFrownRight"]) * 0.5
    t = np.linspace(0, 2 * np.pi, 26)
    mx, my = [], []
    for aa in t:
        x, y = place(np.cos(aa) * wide,
                     -0.34 + np.sin(aa) * open_ + np.cos(aa) ** 2 * smile * 0.08)
        mx.append(x); my.append(y)

    data = [segs[0], segs[1], eyes[0], eyes[1], (mx, my)]
    if artists is None:
        col = "#2b2b2b"
        return [ax.plot(d[0], d[1], c=col,
                        lw=2.0 if i < 2 else 1.6,
                        solid_capstyle="round")[0] for i, d in enumerate(data)]
    for art, d in zip(artists, data):
        art.set_data(d[0], d[1])
    return artists


def draw_face(ax, P, ch, artists=None):
    """Draw brows and a mouth on the head, driven by the face channels.

    The channels were extracted but never rendered, so every sign looked
    expressionless while the video plainly was not. That matters more here than
    on a general-purpose avatar: in Auslan a raised brow marks a yes/no
    question and a lowered one a wh-question, so a face left blank is not
    neutral, it is a missing part of the sign.

    Brow and mouth values are per-signer, already centred on this clip's own
    resting value, so zero means "this signer's neutral" rather than any
    absolute. Head angles are absolute degrees and rotate the whole face.

    Pass `artists` to update an existing set in an animation instead of
    creating new ones each frame.
    """
    hc = head_circle(P)
    if hc is None:
        return artists
    (cx, cy), r = hc
    v = dict(zip(FACE_CH, ch))
    roll = np.radians(v["headRoll"])
    co, si = np.cos(roll), np.sin(roll)

    def place(dx, dy):
        """Face-local offsets in units of head radius -> world, with roll and
        the small shift that yaw and pitch produce on a front view."""
        dx = dx + np.sin(np.radians(v["headYaw"])) * 0.55
        dy = dy + np.sin(np.radians(v["headPitch"])) * 0.55
        return cx + (dx * co - dy * si) * r, cy + (dx * si + dy * co) * r

    lift = np.clip(v["browRaise"] / 0.35, -1, 1) * 0.16
    asym = np.clip(v["browAsym"] / 0.30, -1, 1) * 0.08
    op = np.clip(v["mouthOpen"] / 0.30, -1, 1) * 0.16 + 0.05
    wide = 0.30 + np.clip(v["mouthWide"] / 0.30, -1, 1) * 0.10

    brows, mouth = [], []
    for side, tilt in ((-1, -asym), (1, asym)):
        x0, y0 = place(side * 0.52, 0.26 + lift + tilt)
        x1, y1 = place(side * 0.16, 0.30 + lift + tilt)
        brows.append(([x0, x1], [y0, y1]))
    t = np.linspace(0, 2 * np.pi, 24)
    mx, my = [], []
    for a in t:
        x, y = place(np.cos(a) * wide, -0.34 + np.sin(a) * max(op, 0.02))
        mx.append(x); my.append(y)
    mouth = (mx, my)

    if artists is None:
        col = "#2b2b2b"
        a1 = ax.plot(*brows[0], c=col, lw=2.0, solid_capstyle="round")[0]
        a2 = ax.plot(*brows[1], c=col, lw=2.0, solid_capstyle="round")[0]
        a3 = ax.plot(mouth[0], mouth[1], c=col, lw=1.8)[0]
        return [a1, a2, a3]
    artists[0].set_data(*brows[0])
    artists[1].set_data(*brows[1])
    artists[2].set_data(mouth[0], mouth[1])
    return artists


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


def rig_colour(i):
    n = RT.NAMES[i]
    return HANDCOL[n[-1]] if RT.SKEL[i][2][0] == "hand" else "#2b2b2b"


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--split", default="Valid")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--step", type=int, default=2,
                    help="use every Nth frame; GIF has no interframe\n                         compression worth the name, so a full-rate\n                         three-panel clip runs to megabytes")
    ap.add_argument("--dpi", type=int, default=62)
    ap.add_argument("--rig", default=None)
    a = ap.parse_args()

    stem = f"{a.clip}_kf_rgb"
    frames = read_frames(f"renders/src/{stem}.mp4")
    d2 = np.load(f"keypoints/{a.split}/{stem}.npz", allow_pickle=False)
    _z = np.load(a.rig or f"renders/rt_{a.clip}.npz")
    rig = _z["P_rig"]
    status = _z["status"] if "status" in _z else None
    lab = json.load(open(f"MM-WLAuslan/labels/{a.split}.json")).get(a.clip, "?")
    H, W = frames[0].shape[:2]
    T = min(len(frames), d2["pose"].shape[0], rig.shape[0])
    print(f"{a.clip} '{lab}'  video {len(frames)}f {W}x{H}  "
          f"landmarks {d2['pose'].shape[0]}f  rig {rig.shape[0]}f  -> using {T}")

    pose, lh, rh = d2["pose"], d2["left_hand"], d2["right_hand"]

    keep = list(range(0, T, a.step))
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.4))
    im = ax[0].imshow(frames[0]); ax[0].set_axis_off(); ax[0].set_title("video", fontsize=11)
    im2 = ax[1].imshow(frames[0]); ax[1].set_axis_off()
    ax[1].set_title("detected landmarks", fontsize=11)
    ov = [ax[1].plot([], [], lw=2.0, c="#39d353")[0] for _ in ARM]
    ovh = [[ax[1].plot([], [], lw=1.4, c=HANDCOL[s])[0] for _ in HAND_EDGES]
           for s in ("L", "R")]
    rl = [ax[2].plot([], [], lw=2.0, c=rig_colour(i), solid_capstyle="round")[0]
          for i, _ in BONES]
    head = Circle((0, 0), 0.0, fill=False, ec="#2b2b2b", lw=2.2)
    ax[2].add_patch(head)
    ax[2].set_aspect("equal"); ax[2].set_axis_off()
    (bx0, bx1), (by0, by1) = view_box(rig)
    ax[2].set_xlim(bx0, bx1); ax[2].set_ylim(by0, by1)
    ax[2].set_title("retargeted rig", fontsize=11)
    fig.suptitle(f"{lab}   ({a.clip})", fontsize=12)
    fig.tight_layout()

    def upd(t):
        im.set_data(frames[t]); im2.set_data(frames[t])
        for ln, (u, v) in zip(ov, ARM):
            p, q = pose[t, P11[u]], pose[t, P11[v]]
            ln.set_data([p[0] * W, q[0] * W], [p[1] * H, q[1] * H]) \
                if np.isfinite([p[0], q[0]]).all() else ln.set_data([], [])
        for lns, hand in zip(ovh, (lh, rh)):
            for ln, (u, v) in zip(lns, HAND_EDGES):
                p, q = hand[t, u], hand[t, v]
                ln.set_data([p[0] * W, q[0] * W], [p[1] * H, q[1] * H]) \
                    if np.isfinite([p[0], q[0]]).all() else ln.set_data([], [])
        Pr = rig[t]
        st = None if status is None else status[t]
        for ln, (i, p) in zip(rl, BONES):
            drawable = np.isfinite(Pr[[i, p]]).all()
            if st is not None and (st[i] == 3 or st[p] == 3):
                drawable = False          # ABSENT: never fabricate a hand
            if drawable:
                ln.set_data([Pr[i, 0], Pr[p, 0]], [-Pr[i, 1], -Pr[p, 1]])
                ln.set_alpha(0.30 if (st is not None and
                                      (st[i] in (1, 2) or st[p] in (1, 2)))
                             else 1.0)
            else:
                ln.set_data([], [])
        hc = head_circle(Pr)
        if hc:
            head.set_center(hc[0]); head.set_radius(hc[1])
        else:
            head.set_radius(0.0)
        return [im, im2] + ov + ovh[0] + ovh[1] + rl + [head]

    out = a.out or f"renders/compare_{a.clip}.gif"
    FuncAnimation(fig, upd, frames=keep, interval=1000 / a.fps).save(
        out, writer=PillowWriter(fps=a.fps), dpi=a.dpi)
    print("wrote", out)


if __name__ == "__main__":
    main()
