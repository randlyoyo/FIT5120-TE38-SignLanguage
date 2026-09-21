#!/usr/bin/env python3
"""End to end for one word: pick the best take, retarget it, render it.

    python make_example.py "WHAT"

Panels are video, the rig, and the face/head channels with a cursor. The
channel strip is there because the head and brow signals are the ones a viewer
cannot check by eye against a stick figure -- a 5-degree head turn is invisible
on a skeleton but is the difference between a statement and a role shift.
"""
import argparse, json, zipfile, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

import compare_video as CV
import retarget as RT
import face_channels as FC

ZIPS = {"Train": "MM-WLAuslan/Train/rgb.zip", "Valid": "MM-WLAuslan/Valid/rgb.zip",
        "Test_STU": "MM-WLAuslan/Test_STU/rgb.zip",
        "Test_ITW": "MM-WLAuslan/Test_ITW/rgb.zip",
        "Test_TED": "MM-WLAuslan/Test_TED/rgb.zip"}
SHOW = [("browRaise", "#c1121f", 0), ("mouthOpen", "#3a86ff", 2),
        ("headYaw", "#2a9d8f", 4), ("headPitch", "#e09f3e", 5)]


def build(word, rank=0, outdir="renders"):
    takes = json.load(open(f"{outdir}/takes.json"))
    if word not in takes:
        raise SystemExit(f"{word!r} not in vocabulary")
    t = takes[word][rank]
    split, stem = t["split"], t["stem"]
    mp4 = Path(outdir) / "src" / f"{stem}.mp4"
    if not mp4.exists():
        mp4.parent.mkdir(parents=True, exist_ok=True)
        z = zipfile.ZipFile(ZIPS[split])
        name = next(n for n in z.namelist() if stem in n)
        with z.open(name) as f, open(mp4, "wb") as o:
            o.write(f.read())

    npz = f"keypoints3d/{split}/{stem}.npz"
    os.system(f"{os.sys.executable} retarget.py {npz} "
              f"--out {outdir}/rt_{stem} >/dev/null 2>&1")
    ch, meta = FC.extract(npz)
    return dict(word=word, split=split, stem=stem, mp4=str(mp4),
                rig=f"{outdir}/rt_{stem}.npz", ch=ch, take=t,
                fps=float(meta.get("fps") or 25.0))


def render(ex, out, fps=15.0, step=2):
    frames = CV.read_frames(ex["mp4"])
    z = np.load(ex["rig"])
    rig, status = z["P_rig"], z["status"]
    ch = ex["ch"]
    T = min(len(frames), rig.shape[0], ch.shape[0])
    keep = list(range(0, T, step))

    fig = plt.figure(figsize=(10.6, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 1.35], wspace=0.18)
    axv, axr, axc = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))
    im = axv.imshow(frames[0]); axv.set_axis_off(); axv.set_title("video", fontsize=10)

    lines = [axr.plot([], [], c=CV.rig_colour(i), lw=2.2,
                      solid_capstyle="round")[0] for i, _ in CV.BONES]
    head = Circle((0, 0), 0.0, fill=False, ec="#2b2b2b", lw=2.0)
    axr.add_patch(head)
    gaze, = axr.plot([], [], c="#e63946", lw=2.0)     # nose direction
    (x0, x1), (y0, y1) = CV.view_box(rig)
    axr.set_xlim(x0, x1); axr.set_ylim(y0, y1)
    axr.set_aspect("equal"); axr.set_axis_off()
    axr.set_title("rig", fontsize=10)

    t_ax = np.arange(T) / ex["fps"]
    for name, col, i in SHOW:
        v = ch[:T, i]
        rng = max(np.ptp(v), 1e-6)
        axc.plot(t_ax, (v - v.min()) / rng, c=col, lw=1.5, label=name)
    cursor = axc.axvline(0, c="#333", lw=1.2)
    axc.set_xlim(0, t_ax[-1]); axc.set_ylim(-0.08, 1.08)
    axc.set_yticks([]); axc.set_xlabel("seconds", fontsize=9)
    axc.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    axc.set_title("face / head channels (each scaled to its own range)", fontsize=9)
    fig.suptitle(f"{ex['word']}   —   {ex['split']}/{ex['stem']}   "
                 f"(both-hands {ex['take']['both']:.2f})", fontsize=11)

    def upd(t):
        im.set_data(frames[t])
        Pr, st = rig[t], status[t]
        for ln, (i, p) in zip(lines, CV.BONES):
            ok = np.isfinite(Pr[[i, p]]).all() and st[i] != 3 and st[p] != 3
            ln.set_data([Pr[i, 0], Pr[p, 0]], [-Pr[i, 1], -Pr[p, 1]]) if ok \
                else ln.set_data([], [])
            ln.set_alpha(0.3 if (ok and (st[i] in (1, 2) or st[p] in (1, 2))) else 1.0)
        hc = CV.head_circle(Pr)
        if hc:
            head.set_center(hc[0]); head.set_radius(hc[1])
            # Yaw and pitch drawn as where the nose points: on a stick figure
            # this is the only way head rotation is visible at all.
            yaw, pitch = np.radians(ch[t, 4]), np.radians(ch[t, 5])
            gaze.set_data([hc[0][0], hc[0][0] + hc[1] * 1.6 * np.sin(yaw)],
                          [hc[0][1], hc[0][1] + hc[1] * 1.6 * np.sin(pitch)])
        cursor.set_xdata([t_ax[t], t_ax[t]])
        return [im, head, gaze, cursor] + lines

    FuncAnimation(fig, upd, frames=keep, interval=1000 / fps).save(
        out, writer=PillowWriter(fps=fps), dpi=62)
    plt.close(fig)
    print(f"  {ex['word']:12s} {ex['split']}/{ex['stem']}  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="+")
    ap.add_argument("--rank", type=int, default=0)
    a = ap.parse_args()
    for w in a.words:
        ex = build(w, a.rank)
        render(ex, f"renders/ex_{w.replace(' ', '_').replace('/', '_')}.gif")
