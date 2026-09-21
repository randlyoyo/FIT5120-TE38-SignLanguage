#!/usr/bin/env python3
"""Side-by-side GIFs: reconstructed rig on the left, the real video on the right.

    python batch_compare.py --risk 150          # the 150 likeliest to be wrong
    python batch_compare.py WHALE BALD MOTHER   # named words
    python batch_compare.py --all               # every word (hours, ~2.6 GB)

Written for checking, so the two panels are the only things on screen and the
frame indices are the same on both sides: the extractor stores one row per
decoded frame and nothing downstream resamples, so panel n is the same instant
in both. The video is shown raw, without landmarks drawn on it -- the question
being asked here is whether the rig matches reality, and an overlay answers the
easier question of whether the rig matches the landmarks.
"""
import argparse
import json
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

import compare_video as CV
import retarget as RT

OUT = Path("renders/compare")
SRC = Path("renders/src")
ZIPS = {"Train": "MM-WLAuslan/Train/rgb.zip", "Valid": "MM-WLAuslan/Valid/rgb.zip",
        "Test_STU": "MM-WLAuslan/Test_STU/rgb.zip",
        "Test_ITW": "MM-WLAuslan/Test_ITW/rgb.zip",
        "Test_TED": "MM-WLAuslan/Test_TED/rgb.zip"}
REST = np.array(json.load(open("renders/rig.json"))["restOffsets"])


def ensure_mp4(split, stem):
    p = SRC / f"{stem}.mp4"
    if not p.exists():
        SRC.mkdir(parents=True, exist_ok=True)
        z = zipfile.ZipFile(ZIPS[split])
        name = next(n for n in z.namelist() if stem in n)
        with z.open(name) as f, open(p, "wb") as o:
            o.write(f.read())
    return p


def one(args):
    # The output directory travels in the job rather than in a module global:
    # macOS spawns worker processes, so a global set in the parent's main() is
    # not there when the child re-imports this module, and every job silently
    # resolved back to the default directory and reported "already exists".
    word, info, fps_out, step, dpi, want_face, outdir = args
    outdir = Path(outdir)
    safe = word.replace("/", "_")
    out = outdir / f"{safe}.gif"
    if out.exists():
        return dict(word=word, skipped=True)
    try:
        a = json.load(open(f"renders/anim/{safe}.json"))
        mp4 = ensure_mp4(a["source"]["split"], a["source"]["stem"])
        frames = CV.read_frames(mp4)
        P, _ = RT.fk(np.array(a["rotations"]), REST, np.array(a["rootPositions"]))
        st = np.array(a["status"])
        # Fade only sustained uncertainty; a one-frame fade is a flicker.
        soft = RT.uncertain_mask(st, min_run=3)
        T = min(len(frames), len(P))
        keep = list(range(0, T, step))

        if want_face:
            fig, (axr, axf, axv) = plt.subplots(
                1, 3, figsize=(9.4, 3.7),
                gridspec_kw={"width_ratios": [1, 0.82, 1.16]})
        else:
            fig, (axr, axv) = plt.subplots(1, 2, figsize=(7.0, 3.7),
                                           gridspec_kw={"width_ratios": [1, 1.16]})
            axf = None
        fig.patch.set_facecolor("white")
        lines = [axr.plot([], [], c=CV.rig_colour(i), lw=2.2,
                          solid_capstyle="round")[0] for i, _ in CV.BONES]
        head = Circle((0, 0), 0.0, fill=False, ec="#2b2b2b", lw=2.0)
        axr.add_patch(head)
        # Prefer measured blendshapes; fall back to the 7-channel
        # approximation for clips that have not been through extract_face.py.
        face = ch = bs = bsn = rpy = None
        fp = Path(f"renders/face/{safe}.json")
        if fp.exists():
            fd = json.load(open(fp))
            bs, bsn, rpy = np.array(fd["blendshapes"]), fd["names"], np.array(fd["headRPY"])
            face = [axr.plot([], [], c="#2b2b2b", lw=2.0 if i < 2 else 1.6,
                             solid_capstyle="round")[0] for i in range(5)]
        elif "face" in a:
            ch = np.array(a["face"])
            face = [axr.plot([], [], c="#2b2b2b", lw=2.0,
                             solid_capstyle="round")[0] for _ in range(3)]
        (x0, x1), (y0, y1) = CV.view_box(P)
        axr.set_xlim(x0, x1); axr.set_ylim(y0, y1)
        axr.set_aspect("equal"); axr.set_axis_off()
        axr.set_title("reconstructed rig", fontsize=10)
        # The close-up is the same drawing at a tighter window, so it cannot
        # disagree with the body view -- it is the body view, zoomed.
        face2 = None
        if axf is not None:
            face2 = [axf.plot([], [], c="#2b2b2b", lw=3.4 if i < 2 else 2.6,
                              solid_capstyle="round")[0] for i in range(5)]
            head2 = Circle((0, 0), 0.0, fill=False, ec="#2b2b2b", lw=3.0)
            axf.add_patch(head2)
            axf.set_aspect("equal"); axf.set_axis_off()
            axf.set_title("face", fontsize=10)
        im = axv.imshow(frames[0]); axv.set_axis_off()
        axv.set_title("video", fontsize=10)
        fig.suptitle(f"{word}   —   {a['source']['split']}/"
                     f"{a['source']['stem'].replace('_kf_rgb','')}", fontsize=11)
        fig.tight_layout()

        def upd(t):
            Pr, s, sf = P[t], st[t], soft[t]
            for ln, (i, p) in zip(lines, CV.BONES):
                ok = np.isfinite(Pr[[i, p]]).all() and s[i] != 3 and s[p] != 3
                if ok:
                    ln.set_data([Pr[i, 0], Pr[p, 0]], [-Pr[i, 1], -Pr[p, 1]])
                    ln.set_alpha(0.32 if (sf[i] or sf[p]) else 1.0)
                else:
                    ln.set_data([], [])
            hc = CV.head_circle(Pr)
            if hc:
                head.set_center(hc[0]); head.set_radius(hc[1])
            else:
                head.set_radius(0.0)
            if face is not None:
                if bs is not None and t < len(bs):
                    CV.draw_face_bs(axr, Pr, bs[t], bsn, rpy[t], face)
                elif ch is not None and t < len(ch):
                    CV.draw_face(axr, Pr, ch[t], face)
            if axf is not None:
                hcf = CV.head_circle(Pr)
                if hcf:
                    (fcx, fcy), fr = hcf
                    head2.set_center((fcx, fcy)); head2.set_radius(fr)
                    axf.set_xlim(fcx - fr * 1.7, fcx + fr * 1.7)
                    axf.set_ylim(fcy - fr * 1.7, fcy + fr * 1.7)
                    if bs is not None and t < len(bs):
                        CV.draw_face_bs(axf, Pr, bs[t], bsn, rpy[t], face2)
                    elif ch is not None and t < len(ch):
                        CV.draw_face(axf, Pr, ch[t], face2[:3])
                else:
                    head2.set_radius(0.0)
            im.set_data(frames[t])
            extra = ([head2] + face2) if axf is not None else []
            return lines + [head, im] + (face or []) + extra

        outdir.mkdir(parents=True, exist_ok=True)
        FuncAnimation(fig, upd, frames=keep, interval=1000 / fps_out).save(
            out, writer=PillowWriter(fps=fps_out), dpi=dpi)
        plt.close(fig)
        return dict(word=word, kb=round(out.stat().st_size / 1024))
    except Exception as e:
        return dict(word=word, error=f"{type(e).__name__}: {e}")


def risk_order():
    man = json.load(open("renders/manifest.json"))["words"]
    takes = json.load(open("renders/takes.json"))
    rows = []
    for w in man:
        ts = takes[w["word"]]; t = ts[0]
        sp = np.array([x["spread"] for x in ts])
        good = (sp > 1.4) & (sp < 3.0)
        peer = float(sp[good].max()) if good.any() else None
        r = w["absent"] * 5
        if t["spread"] > 3.0 or t["spread"] < 1.2:
            r += 1.5
        if peer and t["spread"] < peer - 0.5:
            r += peer - t["spread"]
        if t["wristAttach"] > 0.13:
            r += (t["wristAttach"] - 0.13) * 8
        r += max(0.0, 0.60 - t["both"]) * 2
        rows.append((r, w["word"]))
    rows.sort(reverse=True)
    return [w for _, w in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*")
    ap.add_argument("--risk", type=int, default=0, help="the N riskiest words")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fps", type=float, default=14.0)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=64,
                    help="lower makes a smaller GIF; 4.6 GB of them is more "
                         "than anyone can be sent")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--face", action="store_true",
                    help="add a head close-up panel; the head is about 25 px "
                         "tall in the body view, which is far too small to "
                         "judge a brow shape")
    a = ap.parse_args()

    if a.all:
        words = [w["word"] for w in json.load(open("renders/manifest.json"))["words"]]
    elif a.risk:
        words = risk_order()[:a.risk]
    else:
        words = a.words
    if not words:
        raise SystemExit("give word names, --risk N, or --all")

    outdir = Path(a.out) if a.out else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(w, None, a.fps, a.step, a.dpi, a.face, str(outdir)) for w in words]
    print(f"{len(jobs)} words -> {outdir}", flush=True)
    ok = skip = 0
    bad = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            if "error" in r:
                bad.append(r)
            elif r.get("skipped"):
                skip += 1
            else:
                ok += 1
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(jobs)}  ok {ok}  skipped {skip}  failed {len(bad)}",
                      flush=True)
    print(f"\n{ok} written, {skip} already there, {len(bad)} failed")
    for r in bad[:6]:
        print("  ", r["word"], r["error"])
    if outdir.exists():
        tot = sum(p.stat().st_size for p in outdir.glob("*.gif"))
        print(f"{len(list(outdir.glob('*.gif')))} gifs, {tot/1e6:.0f} MB in {outdir}")


if __name__ == "__main__":
    main()
