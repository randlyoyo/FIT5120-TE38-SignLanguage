#!/usr/bin/env python3
"""Turn a raw keypoint .npz into the frame sequence the DTW baseline compares.

The whole point of this file is to strip everything that is a property of the
*camera* while keeping everything that is a property of the *signer*:

    camera -> removed    frame rate, where the person stands, how far away
    signer -> kept       how fast they sign, how big their gestures are
                         relative to their own body, one hand vs two

Order matters. Resampling has to happen before trimming (the motion threshold
is per-second, not per-frame), and centring has to happen before scaling.

Output is (T, D) float32 with NO NaN -- a missing hand becomes zeros plus a
presence flag, so the distance function needs no special cases.
"""

import numpy as np

import slr_common as C

TARGET_FPS = 25.0

# Indices into POSE_SUBSET. Built by name so a change to slr_common cannot
# silently shift them.
I_SHOULDER_L = C.POSE_SUBSET.index(11)
I_SHOULDER_R = C.POSE_SUBSET.index(12)
ARM_IDX = [C.POSE_SUBSET.index(i) for i in (11, 12, 13, 14, 15, 16)]

N_ARM = len(ARM_IDX)
N_HAND = C.N_HAND
# arms + both hands, xy, then 2 presence flags
DIM = (N_ARM + 2 * N_HAND) * 2 + 2

FPS_MIN, FPS_MAX = 5.0, 120.0
MIN_FRAMES = 8


class Unusable(Exception):
    """Clip cannot be turned into a feature sequence."""


def _resample_idx(n_src, fps_src, fps_dst=TARGET_FPS):
    """Nearest-neighbour resample onto a common time base.

    Nearest rather than linear on purpose: hand landmarks are all-or-nothing
    per frame, and interpolating across a gap would invent coordinates that
    were never observed. Duplicating a frame is honest; inventing one is not.

    This maps frames onto real time. It does NOT equalise duration -- a fast
    signer still ends up with fewer frames than a slow one, which is the
    variation DTW exists to absorb.
    """
    dur = n_src / fps_src
    n_dst = max(1, int(round(dur * fps_dst)))
    t = (np.arange(n_dst) + 0.5) / fps_dst          # target frame centres, s
    return np.clip(np.round(t * fps_src - 0.5).astype(int), 0, n_src - 1)


def _trim(wrists, fps=TARGET_FPS):
    """Drop the still lead-in and lead-out.

    Clips open and close with the hands resting at the signer's sides. Those
    frames carry no sign, they differ in length between clips, and they eat
    DTW's alignment budget. Measured earlier: hand detection runs 0.83 across
    the middle 60% of a clip but only 0.54 over the outer 40%.

    Returns (start, stop) or None when there is no usable motion.
    """
    T = wrists.shape[0]
    if T < MIN_FRAMES:
        return None
    v = np.linalg.norm(np.diff(wrists, axis=0), axis=-1)     # (T-1, n_wrist)
    v = np.nan_to_num(v, nan=0.0)      # a wrist that vanished contributes no speed
    speed = v.max(axis=1)
    if speed.max() <= 0:
        return None
    # Threshold relative to this clip's own fast frames, so it survives
    # signers who move a lot and signers who barely move.
    thr = 0.15 * np.percentile(speed, 95)
    active = np.flatnonzero(speed > thr)
    if active.size == 0:
        return None
    pad = int(round(0.1 * fps))                              # ~100 ms margin
    start = max(0, active[0] - pad)
    stop = min(T, active[-1] + 2 + pad)
    if stop - start < MIN_FRAMES:
        return None
    return start, stop


def build(npz_path, presence_weight=1.0, target_fps=TARGET_FPS):
    """.npz path -> (T, DIM) float32, free of NaN.

    Raises Unusable when the clip has no pose, no motion, or broken metadata.
    """
    import json

    d = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))

    fps = float(meta.get("fps", 0.0))
    if not (FPS_MIN <= fps <= FPS_MAX):
        # 14 MTV clips report fps=1000. Their frame counts are normal, so the
        # metadata is wrong rather than the video -- but we cannot place them
        # on a time axis, so they are dropped rather than guessed at.
        raise Unusable(f"fps out of range: {fps}")

    pose = d["pose"][:, :, :2].astype(np.float32)
    lh = d["left_hand"][:, :, :2].astype(np.float32)
    rh = d["right_hand"][:, :, :2].astype(np.float32)
    if pose.shape[0] < MIN_FRAMES:
        raise Unusable(f"only {pose.shape[0]} frames")

    # --- undo the anisotropic squash in normalised image coordinates -------
    # MediaPipe divides x by width and y by height *independently*, so a
    # physically square gesture comes out stretched by the frame's aspect
    # ratio. Train is uniformly 512x408, but Test_MTV mixes 1.33, 1.78 and
    # portrait 0.56 -- a 2.2x disagreement in the x:y ratio for the same sign.
    # Scaling x back by w/h restores square units. The shoulder-width divisor
    # applied later is a single scalar and cannot undo this on its own.
    w, h = float(meta.get("width", 0)), float(meta.get("height", 0))
    if w > 0 and h > 0:
        ar = w / h
        pose[:, :, 0] *= ar
        lh[:, :, 0] *= ar
        rh[:, :, 0] *= ar

    idx = _resample_idx(pose.shape[0], fps, target_fps)
    pose, lh, rh = pose[idx], lh[idx], rh[idx]

    # --- centre on the shoulder midpoint (removes where the person stands) ---
    sl, sr = pose[:, I_SHOULDER_L], pose[:, I_SHOULDER_R]
    origin = (sl + sr) / 2.0                                  # (T, 2)
    if np.isnan(origin).all():
        raise Unusable("no shoulders detected")
    # Carry the last known origin across frames where pose dropped out.
    ok = ~np.isnan(origin[:, 0])
    origin = origin[np.maximum.accumulate(np.where(ok, np.arange(len(ok)), 0))]

    # --- scale by shoulder width (removes distance from camera) ---
    # One median for the whole clip, not per frame: a per-frame divisor would
    # also normalise away the signer leaning in and out, which is real motion.
    width = np.linalg.norm(sl - sr, axis=-1)
    scale = float(np.nanmedian(width))
    if not np.isfinite(scale) or scale < 1e-4:
        raise Unusable("degenerate shoulder width")

    arms = (pose[:, ARM_IDX] - origin[:, None, :]) / scale
    lh_n = (lh - origin[:, None, :]) / scale
    rh_n = (rh - origin[:, None, :]) / scale

    # --- trim still lead-in / lead-out, measured on the wrists ---
    span = _trim(arms[:, [4, 5]], target_fps)                 # wrists in ARM_IDX
    if span is None:
        raise Unusable("no motion")
    a, b = span
    arms, lh_n, rh_n = arms[a:b], lh_n[a:b], rh_n[a:b]

    # --- presence flags, then zero-fill ---
    # A missing hand must cost something. Zero-filling alone would make it
    # free-ish; the flag is what actually carries "this hand is not here", and
    # it is why one-handed vs two-handed signs stay distinguishable (6.1% of
    # Train clips are effectively one-handed).
    have_l = (~np.isnan(lh_n[:, 0, 0])).astype(np.float32)
    have_r = (~np.isnan(rh_n[:, 0, 0])).astype(np.float32)
    lh_n = np.nan_to_num(lh_n, nan=0.0)
    rh_n = np.nan_to_num(rh_n, nan=0.0)
    arms = np.nan_to_num(arms, nan=0.0)

    T = arms.shape[0]
    feat = np.concatenate(
        [
            arms.reshape(T, -1),
            lh_n.reshape(T, -1),
            rh_n.reshape(T, -1),
            (have_l * presence_weight)[:, None],
            (have_r * presence_weight)[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    assert feat.shape[1] == DIM, (feat.shape, DIM)
    if not np.isfinite(feat).all():
        raise Unusable("non-finite feature")
    return feat
