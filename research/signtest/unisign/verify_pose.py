#!/usr/bin/env python3
"""The gate in front of training. Nothing trains until this passes.

Three layers, in increasing order of how much they actually prove:

  1. consistency -- every .npz in the set came from the same spec, the same
     pose model, the same rtmlib build and the same person-selection policy.
     This is the check that catches the failure the plan calls the highest
     risk: two datasets extracted slightly differently. It never raises on its
     own and only shows up as bad BLEU weeks later.

  2. numbers -- per-clip quality flags written to CSV, plus an automated mirror
     test.

  3. eyes -- render the skeleton back onto the source video for a sample,
     including the clips the numbers flagged as worst, and look at them.

The mirror test is worth spelling out. In COCO-WholeBody, "left_shoulder"
(index 5) is the *signer's* left, which for a camera facing them appears on the
RIGHT of the image. So mean x(kp 5) > mean x(kp 6). Auslan distinguishes
dominant from non-dominant hand, so a swap is a different sign, not a cosmetic
problem -- and nothing about it raises.

What a reversed inequality means depends on how many clips show it:

  across a whole dataset -- the pipeline flips x somewhere and every clip has
     its hands swapped. The run fails (--max-mirrored-rate, per dataset/subset).

  on scattered clips -- the chosen person has their back to the camera, or is
     turned far enough side-on that the pose model assigns left and right the
     wrong way round, or is not the signer at all (a nearer bystander facing
     away). A video that is itself mirrored does NOT show up here: the person in
     it still faces the camera and the model labels them consistently. Either
     way those clips' hand labels cannot be trusted, so --exclude-out lists them
     for train.py's data.exclude instead of failing the set.

Stored keypoints are frame-normalised and zero-filled where confidence was too
low, matching Uni-Sign's own extraction. "Missing" here therefore means
confidence <= spec.CONF_THRESHOLD, not NaN.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402

L_SHOULDER, R_SHOULDER = 5, 6           # COCO-WholeBody, 0-based
MIN_SYSTEMIC_CLIPS = 5                  # see --max-mirrored-rate

_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
# Body edges as local positions in spec.PART_INDICES['body'], mirroring the
# graph in stgcn_layers/gcn_utils.py.
_BODY_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (3, 5), (5, 7), (4, 6), (6, 8)]


def load_npz(path: Path, with_people: bool = False):
    with np.load(path, allow_pickle=False) as z:
        out = (z["keypoints"], z["scores"], json.loads(str(z["meta"])))
        if with_people:     # per-frame detection count; absent in older files
            out += (z["n_people"] if "n_people" in z.files else None,)
        return out


# --- layer 1: consistency ---------------------------------------------------

_IDENTITY_FIELDS = ("spec_version", "schema_fingerprint", "pose_model",
                    "pose_backend", "rtmlib_version", "person_select",
                    "coordinate_space")


def check_consistency(metas: list[dict]) -> list[str]:
    problems = []
    for field in _IDENTITY_FIELDS:
        seen = Counter(str(m.get(field)) for m in metas)
        if len(seen) > 1:
            detail = ", ".join(f"{v!r}x{n}" for v, n in seen.most_common())
            problems.append(f"{field}: {len(seen)} distinct values -> {detail}")
    stored = {str(m.get("schema_fingerprint")) for m in metas}
    if stored and stored != {spec.SCHEMA_FINGERPRINT}:
        problems.append(
            f"stored fingerprint {stored} != current spec.py "
            f"{spec.SCHEMA_FINGERPRINT!r}; spec.py changed since extraction -- "
            "re-extract or revert"
        )
    return problems


# --- layer 2: numbers -------------------------------------------------------

def clip_stats(kp: np.ndarray, sc: np.ndarray, meta: dict,
               n_people: np.ndarray | None = None) -> dict:
    T = kp.shape[0]
    thr = spec.CONF_THRESHOLD
    row: dict = {
        "uid": meta["uid"], "dataset": meta["dataset"], "subset": meta["subset"],
        "split": meta["split"], "frames": T, "fps": round(meta.get("fps", 0.0), 2),
    }
    # Frames in which the detector saw more than one person. Not a flag -- the
    # largest box is almost always the signer -- but the first thing to check
    # on a clip the mirror test flags.
    row["multi_person_rate"] = (round(float((n_people > 1).mean()), 4)
                                if n_people is not None and len(n_people) else "")
    flags: list[str] = []

    for part in spec.PART_ORDER:
        idx = spec.PART_INDICES[part]
        present = (sc[:, idx] > thr).mean(axis=1) > 0.5
        row[f"{part}_present"] = round(float(present.mean()) if T else 0.0, 4)

    dom = max(row["left_present"], row["right_present"])
    row["dominant_hand_present"] = round(dom, 4)
    if dom < 0.6:
        flags.append("low_dominant_hand")
    if row["body_present"] < 0.8:
        flags.append("low_body")
    if row["face_all_present"] < 0.5:
        flags.append("low_face")

    both_gone = ((sc[:, spec.PART_INDICES["left"]] <= thr).all(axis=1)
                 & (sc[:, spec.PART_INDICES["right"]] <= thr).all(axis=1))
    longest = cur = 0
    for g in both_gone:
        cur = cur + 1 if g else 0
        longest = max(longest, cur)
    row["longest_both_hand_gap"] = int(longest)
    if T and longest > max(5, 0.3 * T):
        flags.append("long_both_hand_gap")

    if not (sc > thr).any():
        flags.append("nothing_detected")
    else:
        # Coordinates are frame-normalised, so anything well outside [0,1] means
        # the signer left the frame.
        conf = sc > thr
        oob = ((kp[..., 0] < -0.05) | (kp[..., 0] > 1.05)
               | (kp[..., 1] < -0.05) | (kp[..., 1] > 1.05)) & conf
        row["out_of_bounds_rate"] = round(float(oob.sum() / max(conf.sum(), 1)), 4)
        if row["out_of_bounds_rate"] > 0.02:
            flags.append("out_of_bounds")

        if T > 1:
            d = np.abs(np.diff(kp, axis=0)).mean(axis=(1, 2))
            frozen = d < 1e-6
            row["frozen_frame_rate"] = round(float(frozen.mean()), 4)
            if frozen.mean() > 0.2:
                flags.append("frozen_frames")
            row["mean_motion"] = round(float(d.mean()), 6)
            if d.mean() < 1e-4:
                flags.append("no_motion")

    ls, rs = kp[:, L_SHOULDER, 0], kp[:, R_SHOULDER, 0]
    ok = (sc[:, L_SHOULDER] > thr) & (sc[:, R_SHOULDER] > thr)
    if ok.sum() >= 5:
        agree = float((ls[ok] > rs[ok]).mean())
        row["shoulder_orientation_ok"] = round(agree, 4)
        if agree < 0.5:
            flags.append("MIRRORED_OR_BACK_VIEW")
    else:
        row["shoulder_orientation_ok"] = ""

    row["flags"] = "|".join(flags)
    return row


# --- layer 3: eyes ----------------------------------------------------------

def render_overlay(npz_path: Path, out_path: Path, max_frames: int = 240) -> None:
    """Draw the stored keypoints back on the source video.

    Left hand orange, right hand blue, deliberately different so a swap is
    visible at a glance rather than requiring you to count fingers.
    """
    import shutil
    import tempfile

    import cv2

    from extract_pose import _VideoSource  # noqa: PLC0415

    kp, sc, meta = load_npz(npz_path)
    w, h = meta["width"], meta["height"]
    pix = kp * np.array([w, h], dtype=np.float32)
    thr = spec.CONF_THRESHOLD
    colours = {"body": (255, 255, 255), "left": (0, 165, 255),
               "right": (255, 128, 0), "face_all": (140, 255, 140)}

    tmpdir = Path(tempfile.mkdtemp(prefix="unisign-overlay-"))
    try:
        with _VideoSource(meta["source"], tmpdir) as src:
            cap = cv2.VideoCapture(str(src.path))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     meta.get("fps") or 25.0, (w, h))
            for t in range(min(kp.shape[0], max_frames)):
                ok, frame = cap.read()
                if not ok:
                    break

                def px(global_idx):
                    if sc[t, global_idx] <= thr:
                        return None
                    p = pix[t, global_idx]
                    return int(p[0]), int(p[1])

                body = spec.PART_INDICES["body"]
                for a, b in _BODY_EDGES:
                    pa, pb = px(body[a]), px(body[b])
                    if pa and pb:
                        cv2.line(frame, pa, pb, colours["body"], 2)
                for hand in ("left", "right"):
                    idx = spec.PART_INDICES[hand]
                    for a, b in _HAND_EDGES:
                        pa, pb = px(idx[a]), px(idx[b])
                        if pa and pb:
                            cv2.line(frame, pa, pb, colours[hand], 2)
                face = spec.PART_INDICES["face_all"]
                for chain in (range(0, 9), range(9, 17)):
                    ch = list(chain)
                    for a, b in zip(ch, ch[1:]):
                        pa, pb = px(face[a]), px(face[b])
                        if pa and pb:
                            cv2.line(frame, pa, pb, colours["face_all"], 1)
                for part in spec.PART_ORDER:
                    for gi in spec.PART_INDICES[part]:
                        p = px(gi)
                        if p:
                            cv2.circle(frame, p, 2, colours[part], -1)

                for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.putText(frame, "L=orange R=blue", (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thick)
                writer.write(frame)
            writer.release()
            cap.release()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--overlay-dir", type=Path, default=None)
    ap.add_argument("--overlay-n", type=int, default=12)
    ap.add_argument("--overlay-flag", default=None,
                    help="draw overlays for clips carrying this flag instead of "
                         "the worst + random sample (e.g. MIRRORED_OR_BACK_VIEW)")
    ap.add_argument("--exclude-out", type=Path, default=None,
                    help="write the uids of mirrored clips here, one per line, "
                         "for train.py's data.exclude")
    ap.add_argument("--max-mirrored-rate", type=float, default=0.02,
                    help="fail if any dataset/subset has a larger share of "
                         "mirrored clips: that is the pipeline, not a few clips")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    files = sorted(args.npz_dir.glob("*.npz"))
    if not files:
        print(f"no .npz under {args.npz_dir}", file=sys.stderr)
        return 2
    print(f"loading {len(files)} clips ...", file=sys.stderr)

    rows, metas, bad = [], [], []
    for f in files:
        try:
            kp, sc, meta, n_people = load_npz(f, with_people=True)
        except Exception as exc:
            bad.append(f"{f.name}: unreadable ({exc})")
            continue
        metas.append(meta)
        r = clip_stats(kp, sc, meta, n_people)
        r["_path"] = str(f)
        rows.append(r)

    print("\n=== layer 1: consistency ===")
    problems = check_consistency(metas) + bad
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
    else:
        m = metas[0]
        print(f"  OK  spec={m['spec_version']} fp={m['schema_fingerprint']}")
        print(f"      model={m['pose_model']} backend={m.get('pose_backend')} "
              f"rtmlib={m.get('rtmlib_version')} person={m.get('person_select')}")

    print("\n=== layer 2: numbers ===")
    flag_counts: Counter[str] = Counter()
    by_subset: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for fl in filter(None, r["flags"].split("|")):
            flag_counts[fl] += 1
        by_subset[f"{r['dataset']}/{r['subset']}"].append(r["dominant_hand_present"])
    print(f"  clips={len(rows)}")
    for key, vals in sorted(by_subset.items()):
        print(f"  {key:<26} n={len(vals):>6}  dominant-hand present "
              f"median={np.median(vals):.3f}")
    if flag_counts:
        for fl, n in flag_counts.most_common():
            pct = 100.0 * n / len(rows)
            mark = "  <-- investigate" if pct > 10 or fl.startswith("MIRROR") else ""
            print(f"  {fl:<26} {n:>6}  ({pct:.1f}%){mark}")
    else:
        print("  no flags raised")

    mirrored = [r for r in rows if "MIRRORED_OR_BACK_VIEW" in r["flags"].split("|")]
    systemic: list[str] = []
    if mirrored:
        print(f"\n=== mirror test: {len(mirrored)} clips with the shoulders "
              "the wrong way round ===")
        per_group = Counter(f"{r['dataset']}/{r['subset']}" for r in mirrored)
        totals = Counter(f"{r['dataset']}/{r['subset']}" for r in rows)
        for key in sorted(totals):
            n, rate = per_group.get(key, 0), per_group.get(key, 0) / totals[key]
            # A pipeline flip mirrors close to 100% of a group. The floor keeps
            # one odd clip in a small pilot group from reading as one.
            bad = rate > args.max_mirrored_rate and n >= MIN_SYSTEMIC_CLIPS
            if bad:
                systemic.append(key)
            print(f"  {key:<26} {n:>6} / {totals[key]:<6} ({100 * rate:.2f}%)"
                  + ("  <-- SYSTEMIC" if bad else ""))
        by_split = Counter(r["split"] for r in mirrored)
        print("  by split: " + ", ".join(f"{s}={n}" for s, n in sorted(by_split.items())))

        def crowded(rs):
            vals = [r["multi_person_rate"] for r in rs if r["multi_person_rate"] != ""]
            return (100.0 * sum(v > 0.1 for v in vals) / len(vals)) if vals else None
        cm, ca = crowded(mirrored), crowded(rows)
        if cm is not None:
            print(f"  second person in view (>10% of frames): mirrored {cm:.0f}% "
                  f"vs all clips {ca:.0f}%"
                  + ("  <- likely the wrong person, not a pose error" if cm > 2 * ca + 10 else ""))

    if systemic:
        print(f"\n  STOP: more than {args.max_mirrored_rate:.0%} mirrored in "
              f"{', '.join(systemic)}. That is the pipeline flipping x, so every")
        print("  clip there has its hands swapped. Do not train on this set.")
    elif mirrored and args.exclude_out is None:
        print("\n  STOP: left and right hand cannot be trusted in these clips. Re-run")
        print("  with --exclude-out and point data.exclude at the list to train without them.")

    if args.exclude_out:
        args.exclude_out.parent.mkdir(parents=True, exist_ok=True)
        with args.exclude_out.open("w") as fh:
            fh.write(f"# verify_pose.py: MIRRORED_OR_BACK_VIEW, {len(mirrored)} "
                     f"of {len(rows)} clips\n")
            fh.writelines(f"{u}\n" for u in sorted(r["uid"] for r in mirrored))
        if not systemic:
            print(f"\n  {len(mirrored)} clips excluded -> {args.exclude_out}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        cols = [k for k in rows[0] if k != "_path"]
        with args.csv.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n  per-clip stats -> {args.csv}")

    if args.overlay_dir:
        print("\n=== layer 3: eyes ===")
        rng = np.random.default_rng(args.seed)
        if args.overlay_flag:
            flagged = [r for r in rows if args.overlay_flag in r["flags"].split("|")]
            picks = [flagged[i] for i in rng.choice(
                len(flagged), size=min(args.overlay_n, len(flagged)),
                replace=False)] if flagged else []
        else:
            ranked = sorted(rows, key=lambda r: r["dominant_hand_present"])
            half = max(1, args.overlay_n // 2)
            picks = ranked[:half]
            rest = [r for r in rows if r not in picks]
            if rest:
                picks += [rest[i] for i in rng.choice(
                    len(rest), size=min(args.overlay_n - len(picks), len(rest)),
                    replace=False)]
        for r in picks:
            dest = args.overlay_dir / f"{r['uid']}.mp4"
            try:
                render_overlay(Path(r["_path"]), dest)
                print(f"  {dest}  (dominant-hand {r['dominant_hand_present']:.2f} "
                      f"{r['flags'] or 'clean'})")
            except Exception as exc:
                print(f"  FAIL overlay {r['uid']}: {exc}")
        print("\n  Watch these. Orange must be on the signer's left hand.")

    return 1 if (problems or systemic or (mirrored and args.exclude_out is None)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
