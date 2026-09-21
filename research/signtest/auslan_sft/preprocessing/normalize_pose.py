#!/usr/bin/env python3
"""Raw keypoints (.npz) -> normalised, gap-filled, resampled pose tensors (.pt).

Input .npz (written by preprocessing/extract_pose.py, and identical to the
format of unisign/extract_pose.py, so the Auslan-Daily extraction on Drive can
be used as-is):
    keypoints (T, 133, 2) float32, divided by [W, H]
    scores    (T, 133)    float32
    meta      JSON string with at least width, height, fps

Normalisation (per clip, NOT per frame)
---------------------------------------
    p_px[t, j]   = (x[t, j] * W,  y[t, j] * H)            undo per-axis frame scaling
    valid[t, j]  = score[t, j] > conf_threshold
    c_t          = (p_px[t, L_SHOULDER] + p_px[t, R_SHOULDER]) / 2   where both valid
    w_t          = || p_px[t, L_SHOULDER] - p_px[t, R_SHOULDER] ||
    c            = median_t c_t          (one 2-vector for the whole clip)
    s            = median_t w_t          (one scalar for the whole clip)
    p_hat[t, j]  = (p_px[t, j] - c) / s                    units: shoulder widths

One translation and one scale per clip keep global motion (a lean, a shift of
the signing space) intact. A per-frame shoulder frame would erase body motion
and turn shoulder-estimation jitter into whole-skeleton jitter. The median
makes c and s robust to frames where one shoulder is mis-detected.
Image y grows downward and is kept that way; the renderer knows.

Missing keypoints
-----------------
  * gaps of <= max_gap_sec inside a joint's track: linear interpolation, and
    the joint counts as valid (it enters the loss);
  * longer internal gaps: linearly interpolated values for input continuity,
    but valid=False (excluded from every loss and metric);
  * leading/trailing gaps: hold the nearest observed value, valid=False;
  * a joint never observed in the clip: NaN, valid=False. The dataset fills
    NaN with the training mean pose.

Resampling
----------
Linear interpolation in time to `target_fps`. A resampled frame's joint is
valid only if both neighbouring source frames were valid.

Also stored, for the differentiable Uni-Sign converter
(slp_utils/unisign_format.py): Uni-Sign's own crop_scale box, computed exactly
as datasets.py::crop_scale does, over the ORIGINAL body keypoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slp_models import skeleton as sk  # noqa: E402
from slp_utils.config import get_by_path, load_config, resolve_path  # noqa: E402

UNISIGN_BODY_WB = np.asarray(sk.BODY_WB, dtype=np.int64)


def unisign_crop_box(keypoints: np.ndarray, scores: np.ndarray, thr: float):
    """Uni-Sign datasets.py::crop_scale box over body joints, frame-normalised units.

    Returns (xs, ys, scale) or None when fewer than 4 confident body joints exist.
    """
    kp = keypoints[:, UNISIGN_BODY_WB, :]
    sc = scores[:, UNISIGN_BODY_WB]
    valid = kp[sc > thr]
    if len(valid) < 4:
        return None
    xmin, xmax = float(valid[:, 0].min()), float(valid[:, 0].max())
    ymin, ymax = float(valid[:, 1].min()), float(valid[:, 1].max())
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return None
    return (xmin + xmax - scale) / 2, (ymin + ymax - scale) / 2, scale


def _runs(bool_1d: np.ndarray):
    """Yield (start, stop) of consecutive True runs."""
    if not bool_1d.any():
        return
    padded = np.concatenate([[False], bool_1d, [False]])
    d = np.diff(padded.astype(np.int8))
    starts, stops = np.where(d == 1)[0], np.where(d == -1)[0]
    yield from zip(starts, stops)


def fill_gaps(pose: np.ndarray, valid: np.ndarray, max_gap: int):
    """pose (T, J, C) with garbage where ~valid. Returns (filled, valid_out)."""
    T, J, _ = pose.shape
    out = pose.astype(np.float32).copy()
    vout = valid.copy()
    t_idx = np.arange(T)
    for j in range(J):
        v = valid[:, j]
        if not v.any():
            out[:, j] = np.nan
            continue
        if v.all():
            continue
        good = t_idx[v]
        for c in range(out.shape[2]):
            # np.interp holds edge values outside [good[0], good[-1]].
            out[:, j, c] = np.interp(t_idx, good, pose[v, j, c])
        for s, e in _runs(~v):
            internal = s > 0 and e < T
            if internal and (e - s) <= max_gap:
                vout[s:e, j] = True
    return out, vout


def resample(pose: np.ndarray, valid: np.ndarray, src_fps: float, dst_fps: float):
    T = pose.shape[0]
    if T == 0 or abs(src_fps - dst_fps) < 1e-6:
        return pose, valid
    n_out = max(1, int(np.floor((T - 1) * dst_fps / src_fps)) + 1)
    u = np.arange(n_out) * (src_fps / dst_fps)
    lo = np.clip(np.floor(u).astype(np.int64), 0, T - 1)
    hi = np.clip(lo + 1, 0, T - 1)
    w = (u - lo).astype(np.float32)[:, None, None]
    out = (1 - w) * pose[lo] + w * pose[hi]
    # NaN (never-observed joints) propagates, which is what we want.
    v = valid[lo] & valid[hi]
    return out.astype(np.float32), v


def normalize_clip(keypoints: np.ndarray, scores: np.ndarray, width: int, height: int,
                   fps: float, ncfg: dict) -> tuple[dict | None, str]:
    """Return (sample, "") or (None, reason)."""
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 2):
        return None, f"bad keypoints shape {keypoints.shape}"
    if scores.shape != keypoints.shape[:2]:
        return None, f"bad scores shape {scores.shape}"
    if width <= 0 or height <= 0:
        return None, "missing frame size in meta"
    thr = float(ncfg.get("conf_threshold", 0.3))
    T0 = keypoints.shape[0]
    if T0 < 2:
        return None, "fewer than 2 frames"
    if not fps or fps <= 0 or not np.isfinite(fps):
        fps = float(ncfg.get("default_source_fps", 25.0))

    idx = sk.WHOLEBODY_INDICES
    kp = keypoints[:, idx, :].astype(np.float64)
    sc = scores[:, idx].astype(np.float64)
    finite = np.isfinite(kp).all(-1)
    valid = (sc > thr) & finite
    # rtmlib can extrapolate far outside the frame for occluded joints.
    margin = float(ncfg.get("out_of_frame_margin", 0.25))
    inside = ((kp > -margin) & (kp < 1 + margin)).all(-1)
    valid &= inside
    px = kp * np.asarray([width, height], dtype=np.float64)

    ls, rs = sk.L_SHOULDER, sk.R_SHOULDER
    both = valid[:, ls] & valid[:, rs]
    shoulder_frac = float(both.mean())
    if shoulder_frac < float(ncfg.get("min_shoulder_frac", 0.5)):
        return None, f"shoulders visible in only {shoulder_frac:.0%} of frames"
    centers = (px[both, ls] + px[both, rs]) / 2
    widths = np.linalg.norm(px[both, ls] - px[both, rs], axis=-1)
    c = np.median(centers, axis=0)
    s = float(np.median(widths))
    if s < float(ncfg.get("min_shoulder_px", 8.0)):
        return None, f"shoulder width {s:.1f}px too small"

    hands = valid[:, sk.PART_SLICES["left"]].any(-1) | valid[:, sk.PART_SLICES["right"]].any(-1)
    if float(hands.mean()) < float(ncfg.get("min_hand_frac", 0.1)):
        return None, f"a hand is visible in only {hands.mean():.0%} of frames"

    box = unisign_crop_box(keypoints, scores, thr)
    if box is None:
        return None, "Uni-Sign crop box undefined (<4 confident body joints)"

    pose = ((px - c) / s).astype(np.float32)
    pose[~valid] = 0.0
    observed = valid.copy()
    max_gap = max(0, int(round(float(ncfg.get("max_gap_sec", 0.2)) * fps)))
    pose, valid = fill_gaps(pose, valid, max_gap)
    tgt_fps = float(ncfg.get("target_fps", 25.0))
    pose, valid = resample(pose, valid, fps, tgt_fps)
    obs_r = resample(np.zeros((T0, 1, 1), np.float32), observed.astype(bool), fps, tgt_fps)[1]

    T = pose.shape[0]
    if T < int(ncfg.get("min_frames", 8)):
        return None, f"only {T} frames after resampling"
    if T > int(ncfg.get("max_frames", 400)):
        return None, f"{T} frames after resampling exceeds max_frames"

    conf = np.where(observed, sc, np.nan)
    with np.errstate(all="ignore"):
        mean_conf = np.nanmean(conf, axis=0)
    sample = {
        "pose": torch.from_numpy(pose),                     # (T, J, 2), NaN = never seen
        "mask": torch.from_numpy(valid),                    # (T, J) bool, enters loss
        "observed": torch.from_numpy(obs_r),                # (T, J) bool, detector saw it
        "length": int(T),
        "fps": tgt_fps,
        "norm": {
            "center_px": [float(c[0]), float(c[1])],
            "scale_px": s,
            "width": int(width), "height": int(height),
            "source_fps": float(fps), "source_frames": int(T0),
            "conf_threshold": thr,
            "formula": "p_hat = (p_px - center_px) / scale_px, p_px = p_norm * [W, H]",
        },
        "unisign_crop": {"xs": float(box[0]), "ys": float(box[1]), "scale": float(box[2])},
        "mean_confidence": torch.from_numpy(np.nan_to_num(mean_conf, nan=0.0).astype(np.float32)),
        "layout": sk.layout_metadata()["fingerprint"],
    }
    return sample, ""


def denormalize(pose: np.ndarray, norm: dict) -> np.ndarray:
    """Signer space -> pixel coordinates of the source frame."""
    c = np.asarray(norm["center_px"], dtype=np.float32)
    return pose * float(norm["scale_px"]) + c


def load_raw_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        kp = np.asarray(z["keypoints"], dtype=np.float32)
        sc = np.asarray(z["scores"], dtype=np.float32)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return kp, sc, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    ann = resolve_path(cfg, d["annotations_csv"])
    raw_dir = resolve_path(cfg, d["raw_pose_dir"])
    out_dir = resolve_path(cfg, d["pose_dir"])
    ncfg = cfg.get("normalization", {})
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ann.is_file():
        raise FileNotFoundError(f"annotations CSV not found: {ann}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw pose directory not found: {raw_dir} (run extract_pose.py first)")

    from preprocessing.make_splits import read_annotations, sample_id  # noqa: E402
    rows = read_annotations(ann, d)
    if args.limit:
        rows = rows[: args.limit]
    ok = skipped = 0
    rejects = []
    for i, row in enumerate(rows):
        sid = sample_id(row, d)
        dst = out_dir / f"{sid}.pt"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        src = raw_dir / f"{sid}.npz"
        if not src.is_file():
            rejects.append((sid, "raw .npz missing"))
            continue
        try:
            kp, sc, meta = load_raw_npz(src)
            sample, why = normalize_clip(kp, sc, int(meta.get("width", 0)), int(meta.get("height", 0)),
                                         float(meta.get("fps", 0) or 0), ncfg)
        except Exception as exc:  # malformed file: record and continue
            sample, why = None, f"unreadable: {type(exc).__name__}: {exc}"
        if sample is None:
            rejects.append((sid, why))
            continue
        sample["uid"] = sid
        sample["text"] = row[d.get("text_column", "text")]
        tmp = dst.with_suffix(".pt.tmp")
        torch.save(sample, tmp)
        tmp.replace(dst)
        ok += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(rows)} processed, {ok} written, {len(rejects)} rejected", flush=True)

    rej_path = out_dir / "rejected.csv"
    with rej_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "reason"])
        w.writerows(rejects)
    print(f"normalised {ok}, already present {skipped}, rejected {len(rejects)} -> {rej_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
