#!/usr/bin/env python3
"""Temporal continuity of what actually gets DRAWN, for all 3,215 words.

    python check_render.py --workers 6

Every defect a reviewer caught in these animations lived in this layer, not in
the data: the canvas flipped vertically, handshapes interpolated through poses
the signer never made, the palm normal turning the wrist over, a one-frame fade
reading as a flicker. Meanwhile the checks in place -- glb round-trip, head
above hips, left shoulder on the right of frame -- all ran in DATA space, where
nothing was wrong.

So this measures the rendered result frame to frame:

    visFlicker  a bone appearing and vanishing again within two frames
    fadeFlicker the uncertainty fade switching on and off just as briefly
    jump        the largest single-frame move of any joint, in shoulder widths
    dirFlip     the largest single-frame turn of a bone's drawn direction

Thresholds are read off the corpus's own distribution rather than guessed, so
"unusual" means unusual for this data. A high score is a candidate, not a
verdict -- a fast sign genuinely moves fast.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import retarget as RT

REST = np.array(json.load(open("renders/rig.json"))["restOffsets"])
BONES = [(i, RT.PARENT[i]) for i in range(len(RT.SKEL)) if RT.PARENT[i] >= 0]
LONG = None          # filled per clip: bones long enough for direction to mean anything


def isolated_runs(flag, max_len=2):
    """Count runs of True no longer than max_len -- the blink pattern."""
    n = 0
    T = flag.shape[0]
    for j in range(flag.shape[1]):
        t = 0
        while t < T:
            if not flag[t, j]:
                t += 1
                continue
            u = t
            while u < T and flag[u, j]:
                u += 1
            if u - t <= max_len:
                n += 1
            t = u
    return n


def one(word):
    safe = word.replace("/", "_")
    try:
        a = json.load(open(f"renders/anim/{safe}.json"))
        P, _ = RT.fk(np.array(a["rotations"]), REST, np.array(a["rootPositions"]))
        st = np.array(a["status"])
        soft = RT.uncertain_mask(st, min_run=3)
        T = P.shape[0]
        # What the renderer projects: (x, -y), scaled by shoulder width so the
        # numbers mean the same thing for a tall and a short signer.
        sw = float(np.nanmedian(np.linalg.norm(
            P[:, RT.IDX["shoulder_L"]] - P[:, RT.IDX["shoulder_R"]], axis=-1)))
        if not np.isfinite(sw) or sw < 1e-6:
            return dict(word=word, error="degenerate shoulder width")
        S = np.stack([P[:, :, 0], -P[:, :, 1]], axis=-1) / sw

        drawn = st != RT.ABSENT
        vis = np.zeros((T, len(BONES)), bool)
        fade = np.zeros((T, len(BONES)), bool)
        for k, (i, p) in enumerate(BONES):
            vis[:, k] = drawn[:, i] & drawn[:, p]
            fade[:, k] = soft[:, i] | soft[:, p]
        # A bone that blinks on, or blinks off, for one or two frames.
        vis_flicker = isolated_runs(vis & ~np.roll(vis, 1, 0), 2) if T > 3 else 0
        vis_flicker = isolated_runs(~vis, 2) + isolated_runs(vis, 2)
        fade_flicker = isolated_runs(fade, 2)

        d = np.linalg.norm(np.diff(S, axis=0), axis=-1)          # (T-1, J)
        vis_j = drawn[1:] & drawn[:-1]
        jump = float(np.nanmax(np.where(vis_j, d, np.nan))) if vis_j.any() else 0.0

        # Drawn direction, but only where direction is something the eye can
        # read. A first pass thresholded bone length at 0.06 shoulder widths,
        # which let every finger phalanx in -- they run 0.03 to 0.09 -- and the
        # median came out at 126 degrees per frame, which is noise, not motion.
        # Two separate quantities are needed:
        #
        #   dirFlip   turn of the LONG bones (upper arm, forearm, spine,
        #             shoulder line), where a 30-degree jerk is plainly visible.
        #             Short bones are excluded, not down-weighted: a second
        #             metric that normalised the swept chord by the bone's own
        #             length saturated at its maximum for every clip and was
        #             removed.
        v = S[:, [i for i, _ in BONES]] - S[:, [p for _, p in BONES]]
        L = np.linalg.norm(v, axis=-1)
        u = v / np.maximum(L, 1e-9)[..., None]
        dot = np.clip((u[1:] * u[:-1]).sum(-1), -1, 1)
        turn = np.degrees(np.arccos(dot))
        long_ = (L > 0.25) & vis
        m = long_[1:] & long_[:-1]
        dir_flip = float(np.nanmax(np.where(m, turn, np.nan))) if m.any() else 0.0

        # A tipSwing metric was tried here and removed. Chord relative to the
        # bone's OWN length saturates at 2.0 for every clip, because some
        # finger stub always flips 180 degrees somewhere; and the quantity the
        # eye actually reads is absolute screen travel, which `jump` already
        # measures. Two failed attempts at this metric, both from normalising
        # by the wrong thing.

        return dict(word=word, frames=T, visFlicker=int(vis_flicker),
                    fadeFlicker=int(fade_flicker), jump=round(jump, 4),
                    dirFlip=round(dir_flip, 1),
                    absent=round(float((st == RT.ABSENT).mean()), 4))
    except Exception as e:
        return dict(word=word, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="renders/render_check.json")
    a = ap.parse_args()
    words = [w["word"] for w in json.load(open("renders/manifest.json"))["words"]]
    print(f"checking {len(words)} words in render space", flush=True)
    with Pool(a.workers) as pool:
        res = pool.map(one, words, chunksize=16)
    ok = [r for r in res if "error" not in r]
    bad = [r for r in res if "error" in r]

    def col(k):
        return np.array([r[k] for r in ok], float)

    print(f"\n{len(ok)} ok, {len(bad)} errored")
    print(f"{'metric':<13s}{'p50':>9s}{'p90':>9s}{'p99':>9s}{'max':>9s}{'flagged':>9s}")
    print("-" * 58)
    thr = {}
    for k, label in (("visFlicker", "visFlicker"), ("fadeFlicker", "fadeFlicker"),
                     ("jump", "jump(sw)"), ("dirFlip", "dirFlip(deg)")):
        c = col(k)
        t = float(np.percentile(c, 99))
        thr[k] = t
        print(f"{label:<13s}{np.percentile(c,50):9.2f}{np.percentile(c,90):9.2f}"
              f"{t:9.2f}{c.max():9.2f}{int((c > t).sum()):9d}")
    for r in ok:
        r["flags"] = [k for k in thr if r[k] > thr[k]]
    flagged = sorted((r for r in ok if r["flags"]),
                     key=lambda r: -len(r["flags"]))
    print(f"\n{len(flagged)} words exceed at least one 99th percentile")
    print(f"{'word':<28s}{'vis':>5s}{'jump':>7s}{'dir':>7s}  flags")
    for r in flagged[:14]:
        print(f"{r['word'][:27]:<28s}{r['visFlicker']:5d}{r['jump']:7.2f}"
              f"{r['dirFlip']:7.0f}  {','.join(r['flags'])}")
    json.dump({"thresholds": thr, "results": ok, "errors": bad},
              open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
