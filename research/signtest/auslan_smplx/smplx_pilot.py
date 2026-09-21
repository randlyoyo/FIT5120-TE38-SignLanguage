"""Auslan-Daily clip -> SMPL-X rotations -> SignSparK LMDB record.

Body comes from NLF (SMPL-X fit on the whole frame), hands from WiLoR (MANO,
run on hand boxes taken from our rtmlib keypoints, so handedness comes from the
already-verified 2D tracking rather than WiLoR's own detector). This module is
pure numpy/scipy so it can be tested without either model.

Conventions (checked against the SignSparK and WiLoR code, not assumed):
  * rotation 6D = first two ROWS of the matrix (SignSparK `_matrix_to_rot6d_np`);
  * WiLoR's MANO is a `smplx.MANOLayer`, which adds no mean hand pose, so its
    hand_pose is absolute and matches SignSparK's `flat_hand_mean=True`;
  * WiLoR predicts a left hand on the mirrored crop, i.e. as a right hand. That
    raw output is SignSparK's "right-hand (WiLoR) convention". The LMDB stores
    the left hand in true SMPL-X-left convention, F @ R @ F with
    F = diag(1, -1, -1), and SignSparK's loader applies the same conjugation to
    get back to the raw WiLoR form (`flip_left_hand=True`);
  * the same raw left-hand global orientation, mirrored into the real camera,
    is M @ R @ M with M = diag(-1, 1, 1) (conjugating by M or by F is the same
    map, since F = -M);
  * body_features holds the 21 SMPL-X body joints (1..21) as local 6D; the
    SignSparK loader keeps joints 12..21 (neck, collars, head, shoulders,
    elbows, wrists). Wrist rotations come from WiLoR's hand orientation
    composed with NLF's elbow: R_wrist_local = G_elbow^T @ R_hand_camera;
  * face_features = jaw 6D (NLF) + 50 expression coefficients (zeros here: no
    face model in the pilot).
"""

from __future__ import annotations

import io
import pickle

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# SMPL-X kinematic tree for the 55 joints (smplx package order).
SMPLX_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
                          20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
                          21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53])
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, JAW = 18, 19, 20, 21, 22
L_HAND = slice(25, 40)
R_HAND = slice(40, 55)
F_FLIP = np.diag([1.0, -1.0, -1.0])     # SignSparK left-hand flip
M_MIRROR = np.diag([-1.0, 1.0, 1.0])    # image mirror about x

# COCO-WholeBody (rtmlib) indices.
COCO_L_SHOULDER, COCO_R_SHOULDER, COCO_L_WRIST, COCO_R_WRIST = 5, 6, 9, 10
COCO_L_HAND = slice(91, 112)
COCO_R_HAND = slice(112, 133)
# COCO-WholeBody / OpenPose hand order: wrist, thumb 1-4, index 1-4, middle, ring, pinky.
# SMPL-X per hand: index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3 (+15 for the right hand),
# fingertips 66-70 (left thumb, index, middle, ring, pinky) and 71-75 (right).
_OP_FROM_SMPLX_LEFT = [20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70]
OP_FROM_SMPLX = {'left': _OP_FROM_SMPLX_LEFT,
                 'right': [21] + [j + 15 if 25 <= j < 40 else j + 5 for j in _OP_FROM_SMPLX_LEFT[1:]]}
# COCO body joint -> SMPL-X joint, for the arm checks.
COCO_TO_SMPLX_ARMS = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21}


# --------------------------------------------------------------------------- rotations
def aa_to_mat(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, np.float64)
    return Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(*aa.shape[:-1], 3, 3)


def mat_to_aa(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, np.float64)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*R.shape[:-2], 3)


def mat_to_6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R)
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def sixd_to_mat(d6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt on rows, identical to SignSparK `_rot6d_to_matrix_np`."""
    d6 = np.asarray(d6, np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


def global_rotations(local: np.ndarray, upto: int = 22) -> np.ndarray:
    """(T, 55, 3, 3) local -> (T, upto, 3, 3) global, via SMPLX_PARENTS."""
    G = np.empty_like(local[:, :upto])
    for j in range(upto):
        p = SMPLX_PARENTS[j]
        G[:, j] = local[:, j] if p < 0 else G[:, p] @ local[:, j]
    return G


# --------------------------------------------------------------------------- time series
def fill_gaps(x: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """Linear interpolation over time of (T, ...) where `valid` is False; ends hold.
    Returns None if nothing is valid."""
    valid = np.asarray(valid, bool)
    if not valid.any():
        return None
    if valid.all():
        return x.copy()
    T = len(x)
    flat = x.reshape(T, -1).astype(np.float64)
    t = np.arange(T)
    out = np.stack([np.interp(t, t[valid], flat[valid, k]) for k in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


def smooth_6d(d6: np.ndarray, window: int = 5, order: int = 2) -> np.ndarray:
    """Savitzky-Golay along time on 6D features, then re-orthonormalise."""
    T = d6.shape[0]
    w = min(window, T if T % 2 else T - 1)
    if w > order + 1:
        d6 = savgol_filter(d6, w, order, axis=0)
    return mat_to_6d(sixd_to_mat(d6))


# --------------------------------------------------------------------------- 2D keypoints
def hand_box(kp_px: np.ndarray, sc: np.ndarray, thr: float = 0.3, min_pts: int = 8):
    """Tight xyxy box around one hand's 21 keypoints, or None if too few are confident.
    WiLoR's own dataset pads it (rescale_factor), as it does for detector boxes."""
    ok = sc > thr
    if ok.sum() < min_pts:
        return None
    p = kp_px[ok]
    x1, y1 = p.min(0)
    x2, y2 = p.max(0)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def segments_from_wrists(kp_px: np.ndarray, sc: np.ndarray, thr: float = 0.3,
                         min_len: int = 4, min_gap: int = 6) -> np.ndarray:
    """Rough per-frame sign segmentation in SignSparK's labels (0 non-sign, 2 start, 1 continuation).

    A frame is active when either wrist is raised above (shoulder line + 1.5 shoulder
    widths); an active run is split at pronounced speed minima (holds between signs).
    A stand-in for FAST: SignSparK only uses it to pick first/middle/last keyframes.
    """
    T = len(kp_px)
    lab = np.zeros(T, np.int32)
    sh_ok = (sc[:, COCO_L_SHOULDER] > thr) & (sc[:, COCO_R_SHOULDER] > thr)
    if not sh_ok.any():
        return lab
    s = np.median(np.linalg.norm(kp_px[sh_ok, COCO_L_SHOULDER] - kp_px[sh_ok, COCO_R_SHOULDER], axis=-1))
    y_sh = np.median((kp_px[sh_ok, COCO_L_SHOULDER, 1] + kp_px[sh_ok, COCO_R_SHOULDER, 1]) / 2)
    raised = np.zeros(T, bool)
    speed = np.zeros(T)
    for w in (COCO_L_WRIST, COCO_R_WRIST):
        ok = sc[:, w] > thr
        raised |= ok & (kp_px[:, w, 1] < y_sh + 1.5 * s)
        pos = fill_gaps(kp_px[:, w], ok)
        if pos is not None:
            v = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=-1) / max(s, 1e-6)]
            speed = np.maximum(speed, v)
    speed = np.convolve(speed, np.ones(3) / 3, mode='same')
    # close short gaps, drop short runs
    active = raised.copy()
    for run in _runs(~active):
        if run[0] > 0 and run[1] < T and run[1] - run[0] <= 3:
            active[run[0]:run[1]] = True
    for a, b in _runs(active):
        if b - a < min_len:
            active[a:b] = False
    for a, b in _runs(active):
        seg = speed[a:b]
        thr_v = 0.35 * np.median(seg) if len(seg) else 0.0
        cuts, last = [], a
        for t in range(a + 1, b - 1):
            if speed[t] < thr_v and speed[t] <= speed[t - 1] and speed[t] <= speed[t + 1] \
                    and t - last >= min_gap and b - t >= min_len:
                cuts.append(t)
                last = t
        # SignSparK's _segment_bounds only opens a segment on a 0 -> sign transition, so a
        # boundary inside a run must be a 0 frame (the hold) followed by a 2.
        lab[a:b] = 1
        lab[a] = 2
        for t in cuts:
            lab[t] = 0
            lab[t + 1] = 2
    return lab


def _runs(mask: np.ndarray):
    """[start, end) of True runs."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[::2], d[1::2]))


# --------------------------------------------------------------------------- assembly
def assemble(nlf_pose_aa: np.ndarray, hands: dict) -> dict:
    """Combine NLF body and WiLoR hands into SMPL-X local rotations and SignSparK features.

    nlf_pose_aa: (T, 55, 3) NLF SMPL-X local axis-angle (joint 0 = global orient).
    hands[side] for side in ('left', 'right'): dict with
        'global_orient' (T, 3, 3), 'hand_pose' (T, 15, 3, 3) raw WiLoR output
        (left = mirrored-crop prediction), 'valid' (T,) bool.
    Returns features for the LMDB, the composed local rotations (T, 55, 3, 3) for QA,
    and per-side coverage.
    """
    T = nlf_pose_aa.shape[0]
    R = aa_to_mat(nlf_pose_aa)                               # (T, 55, 3, 3) local
    G = global_rotations(R, 22)
    out_local = R.copy()
    coverage = {}
    for side, elbow, wrist, hsl in (('left', L_ELBOW, L_WRIST, L_HAND), ('right', R_ELBOW, R_WRIST, R_HAND)):
        h = hands.get(side)
        valid = np.zeros(T, bool) if h is None else np.asarray(h['valid'], bool)
        coverage[side] = float(valid.mean())
        if not valid.any():
            continue                                         # keep NLF's own wrist and fingers
        go, hp = np.asarray(h['global_orient'], np.float64), np.asarray(h['hand_pose'], np.float64)
        if side == 'left':
            go = M_MIRROR @ go @ M_MIRROR                     # mirrored crop -> real camera
            hp = F_FLIP @ hp @ F_FLIP                         # raw (right-hand convention) -> SMPL-X left
        wrist_loc = np.swapaxes(G[:, elbow], -1, -2) @ go    # G_elbow^T @ R_hand_camera
        # gaps: interpolate in 6D between detected frames, then orthonormalise
        w6 = fill_gaps(mat_to_6d(wrist_loc), valid)
        h6 = fill_gaps(mat_to_6d(hp), valid)
        out_local[:, wrist] = sixd_to_mat(w6)
        out_local[:, hsl] = sixd_to_mat(h6)

    body6 = smooth_6d(mat_to_6d(out_local[:, 1:22]))
    left6 = smooth_6d(mat_to_6d(out_local[:, L_HAND]))
    right6 = smooth_6d(mat_to_6d(out_local[:, R_HAND]))
    jaw6 = smooth_6d(mat_to_6d(out_local[:, JAW]))
    # write the smoothed rotations back so QA sees exactly what goes into the LMDB
    out_local[:, 1:22] = sixd_to_mat(body6)
    out_local[:, L_HAND] = sixd_to_mat(left6)
    out_local[:, R_HAND] = sixd_to_mat(right6)
    out_local[:, JAW] = sixd_to_mat(jaw6)
    feats = {
        'left_features': left6.reshape(T, 90).astype(np.float32),
        'right_features': right6.reshape(T, 90).astype(np.float32),
        'body_features': body6.reshape(T, 126).astype(np.float32),
        'face_features': np.concatenate([jaw6.reshape(T, 6), np.zeros((T, 50))], 1).astype(np.float32),
    }
    return {'features': feats, 'local': out_local, 'coverage': coverage}


# --------------------------------------------------------------------------- LMDB
def record_bytes(language: str, text: str, segment: np.ndarray, feats: dict) -> bytes:
    """One clip in SignSparK's LMDB schema (DATA.md section 2)."""
    buf = io.BytesIO()
    np.savez(buf, language=np.array([language], dtype=object), translation=np.array([text], dtype=object),
             gloss=np.array([''], dtype=object), segment=np.asarray(segment, np.int32), **feats)
    return buf.getvalue()


def write_lmdb(path: str, records: list[tuple[str, bytes]], map_size: int = 1 << 34) -> None:
    import lmdb
    env = lmdb.open(path, map_size=map_size)
    with env.begin(write=True) as txn:
        for key, blob in records:
            txn.put(key.encode(), blob)
        txn.put(b'__meta__', pickle.dumps({'clip_ids': [k for k, _ in records], 'num_clips': len(records)}))
    env.close()


# --------------------------------------------------------------------------- MANO without chumpy
class _ChumpyStub:
    """Stands in for chumpy classes while unpickling MANO_*.pkl (chumpy does not install on
    current numpy/Python); a chumpy array keeps its values in state['x']."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__['_state'] = state


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('chumpy'):
            return _ChumpyStub
        return super().find_class(module, name)


def mano_pkl_to_npz(pkl_path: str, npz_path: str) -> list[str]:
    """Convert MANO_RIGHT.pkl to an .npz smplx can load with ext='npz'. Returns the keys.

    The official MANO_RIGHT.pkl stores some arrays as chumpy *operations* (e.g. a
    Select over another array, with state {'a', 'idxs'} and no 'x'), which only real
    chumpy can evaluate. So real chumpy is used when importable (the current source
    at github.com/mattloper/chumpy imports fine on numpy 2 without installing); the
    stub only handles plain chumpy arrays.
    """
    try:
        import chumpy  # noqa: F401
        with open(pkl_path, 'rb') as fh:
            data = pickle.load(fh, encoding='latin1')
    except ImportError:
        with open(pkl_path, 'rb') as fh:
            data = _StubUnpickler(fh, encoding='latin1').load()
    out = {}
    for k, v in data.items():
        if isinstance(v, _ChumpyStub):
            if 'x' not in v._state:
                raise ValueError(f'{k}: chumpy operation, not a plain array; put the chumpy source on sys.path')
            v = v._state['x']
        elif hasattr(v, 'r') and type(v).__module__.startswith('chumpy'):
            v = v.r
        if hasattr(v, 'toarray'):                            # scipy.sparse (J_regressor)
            v = v.toarray()
        out[k] = np.asarray(v) if not isinstance(v, str) else np.array(v)
    np.savez(npz_path, **out)
    return sorted(out)
