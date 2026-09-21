"""Pose-sequence metrics (numpy, one sequence pair at a time).

Units are shoulder widths (signer space). Generated and reference sequences
generally differ in length, so frame-wise errors are reported two ways:
  * along the DTW alignment path (the usual SLP protocol, e.g. Progressive
    Transformers' DTW-MJE), and
  * after linearly resampling the prediction to the reference length.

What these numbers do NOT tell you: whether the output is Auslan. A skeleton
that hovers near the average signing pose scores a respectable MPJPE; a
correct sign performed at a different place, speed or size by a different
body scores badly. `motion_energy_ratio` and the mean-pose baseline in
evaluate.py exist to expose the first failure; nothing here can detect the
second. Semantic correctness needs back-translation or human (Deaf signer)
judgement.
"""

from __future__ import annotations

import numpy as np

from slp_models import skeleton as sk


def frame_distance_matrix(pred: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray) -> np.ndarray:
    """D[i, j] = mean over GT-valid joints of ||pred_i - gt_j||."""
    d = np.linalg.norm(pred[:, None] - gt[None], axis=-1)            # (Tp, Tg, J)
    v = gt_valid[None].astype(np.float64)
    return (d * v).sum(-1) / np.maximum(v.sum(-1), 1.0)


def dtw(cost: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row, prev = acc[i], acc[i - 1]
        c = cost[i - 1]
        # up/diag are vectorised; left must be a running pass.
        best = np.minimum(prev[1:], prev[:-1])
        for j in range(1, m + 1):
            row[j] = c[j - 1] + min(best[j - 1], row[j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
    path.reverse()
    return float(acc[n, m]), path


def resample_to(x: np.ndarray, T: int) -> np.ndarray:
    if x.shape[0] == T:
        return x
    u = np.linspace(0, x.shape[0] - 1, T)
    lo = np.floor(u).astype(int); hi = np.minimum(lo + 1, x.shape[0] - 1)
    w = (u - lo)[:, None, None]
    return (1 - w) * x[lo] + w * x[hi]


def _part_err(err: np.ndarray, valid: np.ndarray, sl) -> float:
    e, v = err[:, sl], valid[:, sl]
    return float(e[v].mean()) if v.any() else float("nan")


def sequence_metrics(pred: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray,
                     pck_thresholds=(0.1, 0.2)) -> dict:
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[1:] != gt.shape[1:]:
        raise ValueError(f"pred {pred.shape} / gt {gt.shape} must be (T, J, C) with equal J, C")
    pred = np.nan_to_num(pred.astype(np.float64))
    gt = np.nan_to_num(gt.astype(np.float64))
    cost = frame_distance_matrix(pred, gt, gt_valid)
    total, path = dtw(cost)
    pi = np.asarray([p for p, _ in path]); gi = np.asarray([g for _, g in path])
    err = np.linalg.norm(pred[pi] - gt[gi], axis=-1)                 # (P, J)
    val = gt_valid[gi]
    out = {
        "dtw_distance": total / len(path),
        "mpjpe_dtw": float(err[val].mean()) if val.any() else float("nan"),
        "mpjpe_dtw_body": _part_err(err, val, sk.PART_SLICES["body"]),
        "mpjpe_dtw_hands": float(np.nanmean([_part_err(err, val, sk.PART_SLICES["left"]),
                                             _part_err(err, val, sk.PART_SLICES["right"])])),
        "mpjpe_dtw_face": _part_err(err, val, sk.PART_SLICES["face_all"]),
    }
    for a in pck_thresholds:
        out[f"pck@{a}"] = float((err[val] < a).mean()) if val.any() else float("nan")
    # wrist-relative hand error: handshape independent of arm placement
    hand_errs = []
    for part, root in (("left", sk.L_HAND_ROOT), ("right", sk.R_HAND_ROOT)):
        sl = sk.PART_SLICES[part]
        pr = pred[pi][:, sl] - pred[pi][:, root:root + 1]
        gr = gt[gi][:, sl] - gt[gi][:, root:root + 1]
        v = val[:, sl] & val[:, root:root + 1]
        if v.any():
            hand_errs.append(np.linalg.norm(pr - gr, axis=-1)[v].mean())
    out["hand_shape_error_dtw"] = float(np.mean(hand_errs)) if hand_errs else float("nan")

    pr = resample_to(pred, gt.shape[0])
    e2 = np.linalg.norm(pr - gt, axis=-1)
    out["mpjpe_resampled"] = float(e2[gt_valid].mean()) if gt_valid.any() else float("nan")
    if gt.shape[0] >= 2:
        vp, vg = np.diff(pr, axis=0), np.diff(gt, axis=0)
        vv = gt_valid[1:] & gt_valid[:-1]
        out["velocity_error"] = float(np.linalg.norm(vp - vg, axis=-1)[vv].mean()) if vv.any() else float("nan")
        e_p = np.linalg.norm(np.diff(pred, axis=0), axis=-1).mean() if pred.shape[0] >= 2 else 0.0
        e_g = np.linalg.norm(vg, axis=-1)[vv].mean() if vv.any() else float("nan")
        out["motion_energy_ratio"] = float(e_p / e_g) if e_g and np.isfinite(e_g) and e_g > 0 else float("nan")
    out["length_ratio"] = pred.shape[0] / gt.shape[0]
    return out


def aggregate(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float))})
    return {k: float(np.nanmean([r[k] for r in rows if k in r])) for k in keys}
