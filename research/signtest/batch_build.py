#!/usr/bin/env python3
"""Retarget the best take of every word onto one shared skeleton.

    python batch_build.py --rig-only          # build renders/rig.json
    python batch_build.py --workers 6         # then every word

Two things this does that the single-clip path does not:

* One canonical skeleton. retarget.py sizes bones from the clip it is given,
  which is right when inspecting that clip and wrong for a dictionary: the
  avatar would change proportions with every word. Rest offsets are taken as
  the median over a sample of clips and then every word is solved against
  them, so only the rotations vary.
* Split outputs. The rig -- joint names, parents, rest offsets -- is identical
  for all 3,215 words, so it is written once and the per-word files carry only
  what actually differs.

Rotations are rounded to three decimals. On a unit quaternion that is about a
tenth of a degree, far below what is visible, and it cuts the payload roughly
in half against the six decimals a default dump would write.
"""
import argparse
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import face_channels as FC
import retarget as RT

OUT = Path("renders")
ANIM = OUT / "anim"


def canonical_rest(n=250, seed=0):
    """Median rest offsets over a sample of clips -> one skeleton for all words."""
    takes = json.load(open(OUT / "takes.json"))
    rng = random.Random(seed)
    picks = rng.sample(sorted(takes), min(n, len(takes)))
    acc = []
    for w in picks:
        t = takes[w][0]
        p = f"keypoints3d/{t['split']}/{t['stem']}.npz"
        if not os.path.exists(p):
            continue
        try:
            P, _ = RT.positions(p)
            acc.append(RT.rest_from(P, RT.world_frames(P)))
        except Exception:
            continue
    A = np.stack(acc)
    rest = np.nanmedian(A, axis=0)

    # Force the skeleton to be bilaterally symmetric.
    #
    # A human's left and right forearms are the same length. The measured ones
    # are not: over 150 clips the median left forearm comes out 3.7 cm shorter
    # than the right (23.1 against 26.8), an asymmetry already present in the
    # raw world landmarks and widened by the 2D/3D mix. Whatever its cause,
    # baking it into a shared rig gives every one of the 3,215 avatars one
    # short arm, and a lopsided figure reads as a broken reconstruction.
    #
    # Only the LENGTH is averaged, not the offset itself: the two sides' rest
    # offsets live in their own local frames and are not mirror images of each
    # other, so averaging the vectors would bend the skeleton rather than
    # even it up.
    for i, name in enumerate(RT.NAMES):
        if not name.endswith("_L"):
            continue
        j = RT.IDX.get(name[:-2] + "_R")
        if j is None:
            continue
        a, b = np.linalg.norm(rest[i]), np.linalg.norm(rest[j])
        if not (np.isfinite(a) and np.isfinite(b) and a > 1e-6 and b > 1e-6):
            continue
        m = (a + b) / 2.0
        rest[i] *= m / a
        rest[j] *= m / b
    print(f"canonical rest from {len(acc)} clips, symmetrised")
    return rest


def one(args):
    word, take, rest = args
    p3 = Path(f"keypoints3d/{take['split']}/{take['stem']}.npz")
    p2 = Path(f"keypoints/{take['split']}/{take['stem']}.npz")
    p = str(p3 if p3.exists() else p2)
    try:
        P, meta = RT.positions(p)
        R = RT.world_frames(P)
        q = RT.locals_from(R)
        q, status = RT.fill_gaps(q, policy="honest")
        import smooth as SM
        fps = float(meta.get("fps") or 25.0)
        q = SM.one_euro(SM.rot_medoid(SM.align_signs(q), 3), fps)
        root = np.nan_to_num(P[:, 0], nan=0.0)
        ch, _ = FC.extract(p)

        name = word.replace("/", "_")
        ANIM.mkdir(parents=True, exist_ok=True)
        payload = {
            "word": word, "fps": fps,
            "source": {"split": take["split"], "stem": take["stem"],
                       "score": round(take["score"], 3),
                       "bothHands": round(take["both"], 3)},
            "rootPositions": np.round(root, 5).tolist(),
            "rotations": np.round(np.nan_to_num(q, nan=0.0), 3).tolist(),
            "status": status.tolist(),
            "faceChannels": FC.CHANNELS,
            "face": np.round(ch, 3).tolist(),
        }
        with open(ANIM / f"{name}.json", "w") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        absent = float((status == 3).mean())
        return dict(word=word, file=f"anim/{name}.json", frames=int(len(root)),
                    fps=fps, absent=round(absent, 4),
                    score=round(take["score"], 3), stem=take["stem"],
                    split=take["split"])
    except Exception as e:
        return dict(word=word, error=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*", help="only rebuild these words")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rig-only", action="store_true")
    a = ap.parse_args()

    rig_path = OUT / "rig.json"
    if a.rig_only or not rig_path.exists():
        rest = canonical_rest()
        json.dump({"joints": RT.NAMES, "parents": RT.PARENT,
                   "restOffsets": np.round(rest, 6).tolist(),
                   "handJoints": {s: [i for i in range(len(RT.SKEL))
                                      if RT.NAMES[i].endswith(f"_{s}")
                                      and RT.SKEL[i][2][0] == "hand"]
                                  for s in ("L", "R")},
                   "statusLegend": {"0": "measured", "1": "interpolated",
                                    "2": "held", "3": "absent"}},
                  open(rig_path, "w"))
        print(f"-> {rig_path}")
        if a.rig_only:
            return
    rest = np.array(json.load(open(rig_path))["restOffsets"])

    takes = json.load(open(OUT / "takes.json"))
    words = a.words or sorted(takes)
    unknown = sorted(set(words) - set(takes))
    if unknown:
        raise SystemExit("unknown word(s): " + ", ".join(unknown))
    if a.limit:
        words = words[:a.limit]
    jobs = [(w, takes[w][0], rest) for w in words]
    print(f"building {len(jobs)} words with {a.workers} workers", flush=True)

    done, bad = [], []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
            (bad if "error" in r else done).append(r)
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(jobs)}  failed {len(bad)}", flush=True)

    done.sort(key=lambda r: r["word"])
    # A targeted rebuild must not discard the manifest entries for the other
    # words.  Merge the regenerated rows into the existing manifest.
    manifest_path = OUT / "manifest.json"
    old = json.load(open(manifest_path)) if manifest_path.exists() else {"words": []}
    rows = {r["word"]: r for r in old.get("words", [])}
    rows.update({r["word"]: r for r in done})
    merged = [rows[w] for w in sorted(rows)]
    json.dump({"rig": "rig.json", "count": len(merged), "words": merged},
              open(manifest_path, "w"), indent=1)
    print(f"\n{len(done)} ok, {len(bad)} failed -> {OUT/'manifest.json'}")
    for r in bad[:5]:
        print("  ", r["word"], r["error"])
    if done:
        ab = np.array([r["absent"] for r in done])
        print(f"absent joint-frames: median {np.median(ab)*100:.1f}%  "
              f"p90 {np.percentile(ab,90)*100:.1f}%")


if __name__ == "__main__":
    main()
