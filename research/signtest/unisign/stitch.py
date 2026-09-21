#!/usr/bin/env python3
"""Arm D: stitch isolated signs into pseudo-continuous sentences, in pose space.

Optional, and gated on what Arms A/B/C say about the OOV rate. If the test
references are full of words the training set never contained, no amount of
tuning fixes it and more continuous data is the only real answer -- of which
there is 45 hours and no more. Synthesising sentences from the 3,215-gloss
isolated vocabulary is the cheap partial substitute.

Be clear about what the output is. The word order is imposed by whatever
sequence is sampled, the grammar is not Auslan grammar, and every non-manual
marker is wrong -- there is no topicalisation, no question inflection, no
prosody, because none of that exists in a citation-form clip. What a stitched
sentence does carry is the lexical form of each sign and a plausible transition
between neighbours, and that is the part that addresses OOV.

Two things matter for it to help rather than hurt:

  Trim the rest frames. A citation clip begins and ends motionless. Concatenate
  them raw and the model learns that signs are separated by long pauses, which
  is the isolated-timing contamination this whole design is built to avoid.

  Interpolate the transition. Cutting from the end of one sign to the start of
  the next teleports the hands. Real signing moves through a transition, and
  the smoothed path is a better approximation than a jump cut.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402
from manifest import read_manifest, write_manifest  # noqa: E402


def motion_profile(xy: np.ndarray) -> np.ndarray:
    """Per-frame mean hand displacement. xy is (T, 133, 3) = x, y, confidence."""
    hands = np.concatenate([spec.PART_INDICES["left"], spec.PART_INDICES["right"]])
    d = np.abs(np.diff(xy[:, hands, :2], axis=0))     # x, y only, not confidence
    return np.concatenate([[0.0], d.mean(axis=(1, 2))])


def trim_rest(xy: np.ndarray, quantile: float = 0.25,
              pad: int = 2) -> tuple[int, int]:
    """Frame range with the leading and trailing motionless run removed.

    The threshold is relative to the clip's own motion, not absolute: clips
    differ in resolution and signer size, and a fixed pixel threshold silently
    trims everything in a small-framed clip and nothing in a large one.
    """
    prof = motion_profile(xy)
    if prof.size < 5 or not np.isfinite(prof).any():
        return 0, xy.shape[0]
    thresh = float(np.quantile(prof, quantile))
    moving = np.flatnonzero(prof > thresh)
    if moving.size == 0:
        return 0, xy.shape[0]
    lo = max(0, int(moving[0]) - pad)
    hi = min(xy.shape[0], int(moving[-1]) + 1 + pad)
    return lo, hi


def transition(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """n interpolated frames carrying pose `a` to pose `b`.

    Cosine easing rather than linear: a linear ramp starts and stops the hands
    instantaneously, which reads as two velocity discontinuities where a jump
    cut had one.
    """
    if n <= 0:
        return np.zeros((0,) + a.shape, dtype=np.float32)
    t = (1 - np.cos(np.linspace(0, np.pi, n + 2)[1:-1])) / 2
    return (a[None] + t[:, None, None] * (b - a)[None]).astype(np.float32)


def stitch(clips: list[np.ndarray], trans_frames: int = 6,
           do_trim: bool = True) -> np.ndarray:
    pieces: list[np.ndarray] = []
    prev_last: np.ndarray | None = None
    for xy in clips:
        lo, hi = trim_rest(xy) if do_trim else (0, xy.shape[0])
        seg = xy[lo:hi]
        if seg.shape[0] < 2:
            seg = xy
        if prev_last is not None:
            pieces.append(transition(prev_last, seg[0], trans_frames))
        pieces.append(seg)
        prev_last = seg[-1]
    return np.concatenate(pieces, axis=0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True,
                    help="source manifest (isolated clips are selected from it)")
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--out-npz-dir", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--n-sentences", type=int, default=5000)
    ap.add_argument("--min-signs", type=int, default=3)
    ap.add_argument("--max-signs", type=int, default=8)
    ap.add_argument("--transition-frames", type=int, default=6)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train",
                    help="split label for the synthetic rows -- keep it 'train'; "
                         "synthetic sentences must never enter val or test")
    args = ap.parse_args(argv)

    if args.split != "train":
        print("refusing: stitched sentences are synthetic and belong only in "
              "the training split. Evaluating on them measures nothing.",
              file=sys.stderr)
        return 2

    rows = [r for r in read_manifest(args.manifest)
            if r["dataset"] == "mmwlauslan" and r["split"] == "train"
            and (args.npz_dir / f"{r['uid']}.npz").exists()]
    if not rows:
        print("no isolated training clips with extracted poses", file=sys.stderr)
        return 2
    # Group by gloss so a sentence samples distinct signs and then one clip
    # each. Sampling clips directly repeats glosses within a sentence, which is
    # rare in real signing and teaches the decoder to stutter.
    by_gloss: dict[str, list[dict]] = {}
    for r in rows:
        by_gloss.setdefault(r["gloss"], []).append(r)
    glosses = sorted(by_gloss)
    print(f"source pool: {len(rows)} isolated clips, {len(glosses)} glosses")
    if len(glosses) < args.max_signs:
        print(f"  note: only {len(glosses)} glosses available, capping "
              f"sentence length at that", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    args.out_npz_dir.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []

    for i in range(args.n_sentences):
        hi = min(args.max_signs, len(glosses))
        k = int(rng.integers(min(args.min_signs, hi), hi + 1))
        chosen = [glosses[j] for j in rng.choice(len(glosses), size=k, replace=False)]
        picks = [by_gloss[g][int(rng.integers(len(by_gloss[g])))] for g in chosen]
        clips, metas = [], []
        for r in picks:
            with np.load(args.npz_dir / f"{r['uid']}.npz", allow_pickle=False) as z:
                clips.append(np.concatenate(
                    [z["keypoints"], z["scores"][..., None]], axis=-1))
                metas.append(json.loads(str(z["meta"])))
        # Every source clip must share the frame size, or the stored pixel
        # coordinates are not on one scale and normalisation at load time
        # produces a sentence assembled from differently-sized signers.
        sizes = {(m["width"], m["height"]) for m in metas}
        if len(sizes) > 1:
            continue
        w, h = sizes.pop()

        xy = stitch(clips, args.transition_frames, not args.no_trim)
        text = " ".join(r["gloss"] for r in picks)
        uid = f"stitch-{i:06d}"
        meta = {
            "uid": uid, "dataset": "stitched", "subset": "pseudo_continuous",
            "split": "train", "text": text, "gloss": None, "camera": "kf",
            "source": "|".join(r["uid"] for r in picks),
            "spec_version": spec.SPEC_VERSION,
            "schema_fingerprint": spec.SCHEMA_FINGERPRINT,
            "pose_model": metas[0]["pose_model"],
            "pose_backend": metas[0].get("pose_backend"),
            "rtmlib_version": metas[0].get("rtmlib_version"),
            "person_select": metas[0].get("person_select"),
            "coordinate_space": "frame_normalised",
            "width": w, "height": h, "fps": metas[0].get("fps", 25.0),
            "n_frames": int(xy.shape[0]),
            "synthetic": True,
        }
        # Stitched clips carry x, y and confidence through together so the
        # transition frames get an interpolated confidence too, rather than a
        # fabricated 1.0 on a pose that was never observed.
        kp, sc = xy[..., :2].astype(np.float32), xy[..., 2].astype(np.float32)
        with (args.out_npz_dir / f"{uid}.npz").open("wb") as fh:
            np.savez_compressed(fh, keypoints=kp, scores=sc,
                                meta=json.dumps(meta))
        out_rows.append({
            "uid": uid, "dataset": "stitched", "subset": "pseudo_continuous",
            "split": "train", "video": "synthetic", "text": text,
            "gloss": None, "camera": "kf",
        })
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{args.n_sentences}", file=sys.stderr)

    write_manifest(out_rows, args.out_manifest)
    print(f"wrote {len(out_rows)} pseudo-continuous sentences -> {args.out_manifest}")
    print("Add them to Arm C's aux set, or concatenate this manifest with the "
          "real one. They are training data only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
