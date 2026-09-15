#!/usr/bin/env python3
"""Turn a 3D keypoint .npz into a viewpoint-canonical frame sequence.

This is the 3D counterpart of dtw_features.py and exists to test one claim:
that Test_MTV collapses because of VIEWPOINT, and that metric world landmarks
can undo the viewpoint where an image-plane transform cannot.

    2D pipeline   camera -> removed   position, distance, frame rate
                  camera -> KEPT      which way the signer faces   <-- the bug
    3D pipeline   camera -> removed   position, distance, frame rate, FACING

Everything else is deliberately identical to the 2D builder -- same fps
resample, same shoulder-width scale, same motion trim, same presence flags --
so a difference in the result is attributable to the canonical rotation and
the extra z channel, and not to a hundred small pipeline changes.

Output is (T, 146) float32, NaN-free:  (6 arm + 42 hand) points x 3 + 2 flags.
"""

import json

import numpy as np

import dtw_features as F2
from dtw_features import Unusable, TARGET_FPS, FPS_MIN, FPS_MAX, MIN_FRAMES

# Raw MediaPipe pose indices. pose_world keeps all 33 points (extract3d.py
# stores the full set precisely so the hips are available here), so these are
# used directly rather than through POSE_SUBSET.
ARM_RAW = [11, 12, 13, 14, 15, 16]      # shoulders, elbows, wrists
I_SH_L, I_SH_R = 11, 12
I_HIP_L, I_HIP_R = 23, 24

N_ARM = len(ARM_RAW)
N_HAND = 21
DIM = (N_ARM + 2 * N_HAND) * 3 + 2       # 146


def _body_frame(pose_w):
    """Per-clip orthonormal body frame -> (3,3) rotation, rows = new axes.

    Built from the torso rather than per frame. A per-frame frame would also
    cancel the signer turning or leaning DURING the sign, which is real
    articulation; one frame for the clip removes only where the camera stood.

        up      hip midpoint -> shoulder midpoint
        right   left shoulder -> right shoulder, orthogonalised against up
        fwd     up x right, the direction the signer faces

    Expressing every point in this basis is what makes an off-axis capture
    comparable with a frontal one.
    """
    mid = lambda a, b: (pose_w[:, a] + pose_w[:, b]) / 2.0
    up = np.nanmedian(mid(I_SH_L, I_SH_R) - mid(I_HIP_L, I_HIP_R), axis=0)
    right = np.nanmedian(pose_w[:, I_SH_R] - pose_w[:, I_SH_L], axis=0)
    if not (np.isfinite(up).all() and np.isfinite(right).all()):
        raise Unusable("no torso")
    nu = np.linalg.norm(up)
    if nu < 1e-4:
        raise Unusable("degenerate torso")
    up = up / nu
    right = right - up * float(right @ up)          # Gram-Schmidt
    nr = np.linalg.norm(right)
    if nr < 1e-4:
        # Shoulders parallel to the spine: the azimuth is undefined and any
        # rotation we picked would be arbitrary, so refuse rather than guess.
        raise Unusable("degenerate shoulder axis")
    right = right / nr
    fwd = np.cross(up, right)
    return np.stack([right, up, fwd]).astype(np.float32)


def build(npz_path, presence_weight=1.0, target_fps=TARGET_FPS):
    """.npz path -> (T, DIM) float32, free of NaN. Raises Unusable."""
    d = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))

    fps = float(meta.get("fps", 0.0))
    if not (FPS_MIN <= fps <= FPS_MAX):
        raise Unusable(f"fps out of range: {fps}")

    pose = d["pose_world"].astype(np.float32)            # (T,33,3) metres
    lh = d["left_hand_world"].astype(np.float32)         # (T,21,3), body frame
    rh = d["right_hand_world"].astype(np.float32)
    if pose.shape[0] < MIN_FRAMES:
        raise Unusable(f"only {pose.shape[0]} frames")

    # No aspect-ratio correction here: world landmarks are metric and already
    # isotropic. The 2D builder's w/h fix has no counterpart and needs none.

    idx = F2._resample_idx(pose.shape[0], fps, target_fps)
    pose, lh, rh = pose[idx], lh[idx], rh[idx]

    R = _body_frame(pose)                                # (3,3)

    # --- centre on the shoulder midpoint, same landmark the 2D builder uses ---
    origin = (pose[:, I_SH_L] + pose[:, I_SH_R]) / 2.0   # (T,3)
    if np.isnan(origin).all():
        raise Unusable("no shoulders detected")
    ok = ~np.isnan(origin[:, 0])
    origin = origin[np.maximum.accumulate(np.where(ok, np.arange(len(ok)), 0))]

    # --- scale by shoulder width, one median for the clip ---
    width = np.linalg.norm(pose[:, I_SH_L] - pose[:, I_SH_R], axis=-1)
    scale = float(np.nanmedian(width))
    if not np.isfinite(scale) or scale < 1e-4:
        raise Unusable("degenerate shoulder width")

    def canon(p):
        return ((p - origin[:, None, :]) / scale) @ R.T

    arms = canon(pose[:, ARM_RAW])
    lh_n, rh_n = canon(lh), canon(rh)

    # --- trim still lead-in / lead-out, measured on the wrists ---
    span = F2._trim(arms[:, [4, 5]], target_fps)         # wrists in ARM_RAW
    if span is None:
        raise Unusable("no motion")
    a, b = span
    arms, lh_n, rh_n = arms[a:b], lh_n[a:b], rh_n[a:b]

    have_l = (~np.isnan(lh_n[:, 0, 0])).astype(np.float32)
    have_r = (~np.isnan(rh_n[:, 0, 0])).astype(np.float32)
    lh_n = np.nan_to_num(lh_n, nan=0.0)
    rh_n = np.nan_to_num(rh_n, nan=0.0)
    arms = np.nan_to_num(arms, nan=0.0)

    T = arms.shape[0]
    feat = np.concatenate(
        [arms.reshape(T, -1), lh_n.reshape(T, -1), rh_n.reshape(T, -1),
         (have_l * presence_weight)[:, None],
         (have_r * presence_weight)[:, None]],
        axis=1).astype(np.float32)

    assert feat.shape[1] == DIM, (feat.shape, DIM)
    if not np.isfinite(feat).all():
        raise Unusable("non-finite feature")
    return feat
