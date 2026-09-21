#!/usr/bin/env python3
"""Rank every take of every word, and pick the best one to drive an avatar.

    python select_takes.py --out renders/takes.json

The corpus holds about a dozen recordings of each of the 3,215 words. Measured
over Train, a randomly chosen take has both hands detected in 0.68 of frames
while the best take of the same word reaches 0.92, and 95.3% of words have a
take at or above 0.80. So the fix for an occluded hand is not to reconstruct
it -- it is to use one of the eleven other recordings where it was not
occluded.

Detection rate alone is not enough to call a take good, so four other things
that break a teaching clip are scored too. All are read from the 3D .npz,
which also carries the normalised 2D arrays, so each clip is opened once.

Test_MTV is excluded: it is off-axis multi-view footage, which is exactly what
an avatar shown head-on should not be built from. Test_SYN is excluded as
synthetic.
"""
import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import slr_common as C

P11 = {v: i for i, v in enumerate(C.POSE_SUBSET)}
SPLITS = ["Train", "Valid", "Test_STU", "Test_ITW", "Test_TED"]
SPAN_BAND = (0.177, 0.257)      # measured p2-p98 of the corpus's own framing


def score_clip(args):
    split, path = args
    stem = os.path.basename(path)[:-4]
    try:
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        w, h = meta.get("width", 0), meta.get("height", 0)
        if not (w and h):
            return None
        lhw, rhw = d["left_hand_world"], d["right_hand_world"]
        lh = ~np.isnan(lhw[:, 0, 0])
        rh = ~np.isnan(rhw[:, 0, 0])
        both = float((lh & rh).mean())

        # Longest unbroken absence, not the overall rate. Two takes can lose
        # the same fraction of frames and be worlds apart: scattered single
        # frames interpolate away invisibly, while one long dropout cannot be
        # bridged at all and has to be rendered as a missing hand. The rate
        # alone ranked a take with 17.6% unrenderable frames above one that
        # found both hands 98% of the time.
        gaps = []
        for k in ("left_hand_world", "right_hand_world"):
            miss = np.isnan(d[k][:, 0, 0])
            run = best = 0
            for m_ in miss:
                run = run + 1 if m_ else 0
                best = max(best, run)
            gaps.append(best)
        max_gap = int(max(gaps)) if gaps else 0
        fps_ = float(meta.get("fps") or 25.0)
        # retarget.fill_gaps bridges up to 10 frames at 25 fps, so express the
        # limit in seconds and score against that rather than a frame count
        # that would mean different things at 15 and 60 fps.
        gap_s = max_gap / max(fps_, 1.0)
        f_gap = float(np.clip(1.0 - gap_s / 0.5, 0, 1))

        # in-frame: a hand touching the edge is a hand we cannot show
        inside = []
        for k in ("left_hand", "right_hand"):
            a = d[k][:, :, :2]
            ok = ~np.isnan(a[:, 0, 0])
            if ok.sum() == 0:
                continue
            b = a[ok]
            out = ((b[..., 0] < 0.02) | (b[..., 0] > 0.98) |
                   (b[..., 1] < 0.02) | (b[..., 1] > 0.98)).any(-1)
            inside.append(1.0 - float(out.mean()))
        inframe = float(np.mean(inside)) if inside else 0.0

        # framing: shoulder span against the corpus's own distribution
        p = d["pose"][:, :, :2].astype(np.float32).copy()
        p[:, :, 0] *= w / h
        span = float(np.nanmedian(np.linalg.norm(
            p[:, P11[11]] - p[:, P11[12]], axis=-1)))
        lo, hi = SPAN_BAND
        framing = 1.0 if lo <= span <= hi else max(
            0.0, 1.0 - abs(span - np.clip(span, lo, hi)) / (hi - lo))

        # --- correctness, not just presence -------------------------------
        # Detection rate says a hand was found, never that it was found in the
        # right place or the right shape. Two failures slipped through on rate
        # alone: a take whose finger landmarks collapse onto the palm (the hand
        # is "detected" at 0.98 but every handshape reads as a fist), and one
        # whose hand sits away from the wrist the pose reports.
        p2 = d["pose"][:, :, :2].astype(np.float64).copy()
        p2[:, :, 0] *= w / h
        sw2 = np.nanmedian(np.linalg.norm(
            p2[:, P11[11]] - p2[:, P11[12]], axis=-1))
        smid = (p2[:, P11[11], 1] + p2[:, P11[12], 1]) / 2
        attach, spread_ = [], []
        for k, pi in (("left_hand", 15), ("right_hand", 16)):
            a = d[k][:, :, :2].astype(np.float64).copy()
            a[:, :, 0] *= w / h
            ok = ~np.isnan(a[:, 0, 0])
            if ok.sum() < 5 or not np.isfinite(sw2) or sw2 < 1e-6:
                continue
            attach.append(float(np.nanmedian(
                np.linalg.norm(a[ok, 0] - p2[ok, P11[pi]], axis=-1) / sw2)))
            # Fingers only extend when the hand is up and working, so the
            # measure is taken on this clip's own highest 40% of frames rather
            # than against an absolute height -- some signs never break the
            # shoulder line at all.
            lift = (smid - p2[:, P11[pi], 1]) / sw2
            sel = ok & (lift >= np.nanpercentile(lift[ok], 60))
            if sel.sum() < 3:
                continue
            palm = np.linalg.norm(a[:, 9] - a[:, 0], axis=-1)
            e = np.stack([np.linalg.norm(a[:, t] - a[:, 0], axis=-1) / palm
                          for t in (8, 12, 16)], axis=1)
            spread_.append(float(np.nanpercentile(np.nanmax(e, axis=1)[sel], 90)))
        # Anatomy puts an extended finger near 1.85 of palm length and the
        # corpus median is 2.30, so 1.4 is where a hand has stopped opening at
        # all. Attachment error runs 0.06 shoulder widths at the median and
        # 0.14 at p95.
        spread = max(spread_) if spread_ else 0.0
        wrist_attach = float(np.mean(attach)) if attach else 1.0
        # Two-sided on purpose. A one-sided version scored a clip whose
        # fingertips measured 20.9 palm-lengths from the wrist as perfect: the
        # palm measurement had collapsed towards zero and the ratio exploded,
        # so the very takes whose hand landmarks were most broken ranked first.
        # A real extended finger is about 1.85 palm-lengths and nothing
        # anatomical exceeds ~3, so beyond that the measurement has failed and
        # the take must be rejected rather than rewarded.
        if spread > 3.0:
            f_spread = 0.0
        else:
            f_spread = float(np.clip((spread - 1.4) / (2.3 - 1.4), 0, 1))
        f_attach = float(np.clip(1.0 - wrist_attach / 0.15, 0, 1))

        # jitter: a rigid skeleton cannot follow a wobbling measurement, so
        # per-frame bone-length variation predicts how well this take will
        # retarget. Measured on the fingers, where it is worst.
        jit = []
        for hw, ok in ((lhw, lh), (rhw, rh)):
            if ok.sum() < 5:
                continue
            b = hw[ok]
            for a_, b_ in ((5, 6), (6, 7), (9, 10), (10, 11), (13, 14)):
                L = np.linalg.norm(b[:, a_] - b[:, b_], axis=-1)
                m = np.median(L)
                if m > 1e-4:
                    jit.append(L.std() / m)
        jitter = float(np.mean(jit)) if jit else 1.0

        # Framing carries no weight: measured over all 57,870 takes it is
        # exactly 1.0 for 95% of them and never below 0.999, so it cannot
        # separate anything. It is still reported, as a check that a future
        # corpus has not drifted out of the band. `inframe` earns its weight
        # only in the worst fifth, where 22% of takes do touch an edge.
        score = (0.30 * both + 0.20 * f_gap + 0.08 * inframe +
                 0.08 * max(0.0, 1.0 - jitter / 0.35) +
                 0.22 * f_spread + 0.12 * f_attach)
        return dict(split=split, stem=stem, both=both, inframe=inframe,
                    framing=framing, jitter=jitter, span=span,
                    spread=spread, wristAttach=wrist_attach,
                    maxGapSec=round(gap_s, 3),
                    frames=int(lhw.shape[0]), fps=meta.get("fps"),
                    score=float(score))
    except Exception as e:
        return dict(split=split, stem=stem, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="renders/takes.json")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    jobs, labels = [], {}
    for sp in SPLITS:
        lab = json.load(open(f"MM-WLAuslan/labels/{sp}.json"))
        labels[sp] = lab
        jobs += [(sp, str(p)) for p in sorted(Path(f"keypoints3d/{sp}").glob("*.npz"))]
    print(f"scoring {len(jobs)} takes across {len(SPLITS)} splits", flush=True)

    with Pool(a.workers) as pool:
        res = pool.map(score_clip, jobs, chunksize=64)

    key = lambda s: s[:-7] if s.endswith("_kf_rgb") else s
    by = defaultdict(list)
    bad = 0
    for r in res:
        if r is None or "error" in r:
            bad += 1
            continue
        w = labels[r["split"]].get(key(r["stem"]))
        if w:
            by[w].append(r)
    for w in by:
        by[w].sort(key=lambda r: -r["score"])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({w: v for w, v in sorted(by.items())}, open(a.out, "w"))
    best = np.array([v[0]["score"] for v in by.values()])
    print(f"{len(by)} words, {bad} unusable takes -> {a.out}")
    print(f"best-take score: median {np.median(best):.3f}  "
          f"p10 {np.percentile(best, 10):.3f}  min {best.min():.3f}")


if __name__ == "__main__":
    main()
