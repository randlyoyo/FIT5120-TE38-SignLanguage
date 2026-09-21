#!/usr/bin/env python3
"""Keypoint positions -> skeleton rotations -> animation, with a round-trip check.

    python retarget.py keypoints3d/Valid/88500_kf_rgb.npz --out /tmp/thankyou

MediaPipe gives joint POSITIONS. An avatar rig is driven by joint ROTATIONS.
This converts one to the other without an IK solver: because every joint the
rig needs already has a measured position, each joint's world frame can be
read straight off the data -- primary axis down the bone, secondary axis from
a reference the anatomy supplies (the arm plane, the palm normal). An
iterative solver would be slower, jitterier and no more correct.

Two things it deliberately does NOT hide:

* Retargeting replaces the signer's bone lengths with the rig's, so joint
  positions cannot be reproduced exactly and should not be. `--check` reports
  both errors separately: rotation-only (against the signer's own lengths,
  which measures whether the SOLVE is right) and full retarget (which measures
  how much the proportion change moved things).
* Hands are absent from 20-37% of frames. Rotations are interpolated across
  short gaps and held across long ones, and every filled frame is flagged in
  the export so a renderer can fade rather than snap.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

import slr_common as C

# --- skeleton -------------------------------------------------------------
# (name, parent, source, aim child, secondary-axis rule)
#
# `source` reads a position: ("pose", i) | ("mid", i, j) | ("hand", side, i).
# `aim` names the child whose direction defines this joint's primary axis; a
# joint with no aim is a leaf and carries no rotation of its own.
FINGERS = [("thumb", 1, 2, 3, 4), ("index", 5, 6, 7, 8), ("middle", 9, 10, 11, 12),
           ("ring", 13, 14, 15, 16), ("pinky", 17, 18, 19, 20)]


def build_skeleton():
    J = [("hips", None, ("mid", 23, 24), "chest", "hipaxis"),
         ("chest", "hips", ("mid", 11, 12), "neck", "shoulderaxis"),
         ("neck", "chest", ("pose", 0), None, None)]
    for side, sh, el, wr in (("L", 11, 13, 15), ("R", 12, 14, 16)):
        J += [(f"shoulder_{side}", "chest", ("pose", sh), f"elbow_{side}", f"armplane_{side}"),
              (f"elbow_{side}", f"shoulder_{side}", ("pose", el), f"wrist_{side}", f"armplane_{side}"),
              (f"wrist_{side}", f"elbow_{side}", ("pose", wr), f"middle1_{side}", f"palm_{side}")]
        for name, a, b, c, d in FINGERS:
            J += [(f"{name}1_{side}", f"wrist_{side}", ("hand", side, a), f"{name}2_{side}", f"palm_{side}"),
                  (f"{name}2_{side}", f"{name}1_{side}", ("hand", side, b), f"{name}3_{side}", f"palm_{side}"),
                  (f"{name}3_{side}", f"{name}2_{side}", ("hand", side, c), f"{name}4_{side}", f"palm_{side}"),
                  (f"{name}4_{side}", f"{name}3_{side}", ("hand", side, d), None, None)]
    return J


SKEL = build_skeleton()
NAMES = [j[0] for j in SKEL]
IDX = {n: i for i, n in enumerate(NAMES)}
PARENT = [IDX[j[1]] if j[1] else -1 for j in SKEL]
AIM = [IDX[j[3]] if j[3] else -1 for j in SKEL]
REF = [j[4] for j in SKEL]
HAND_JOINTS = {"L": [i for i, n in enumerate(NAMES) if n.endswith("_L") and SKEL[i][2][0] == "hand"],
               "R": [i for i, n in enumerate(NAMES) if n.endswith("_R") and SKEL[i][2][0] == "hand"]}


# The 11-point 2D pose subset, by raw MediaPipe index.
_P11 = {v: i for i, v in enumerate(C.POSE_SUBSET)}


def _positions_2d(d):
    """Build the drawable skeleton directly from the 2D extraction.

    The 3D extractor is optional in this repository.  When it is unavailable,
    preserving the measured image-plane handshape is safer than silently
    reusing stale 3D output: the review renderer is a frontal stick figure and
    its visible geometry is the 2D geometry.  Depth is set to zero, so it is
    deliberately not used to claim a palm-facing direction.
    """
    meta = json.loads(str(d["meta"]))
    pose = d["pose"].astype(np.float64)[..., :2].copy()
    hands = {"L": d["left_hand"].astype(np.float64)[..., :2].copy(),
             "R": d["right_hand"].astype(np.float64)[..., :2].copy()}
    w, h = float(meta.get("width", 0)), float(meta.get("height", 0))
    xscale = w / h if w and h else 1.0
    pose[..., 0] *= xscale
    for hand in hands.values():
        hand[..., 0] *= xscale

    T = pose.shape[0]
    P = np.full((T, len(SKEL), 3), np.nan, dtype=np.float64)
    for i, (_, _, src, _, _) in enumerate(SKEL):
        if src[0] == "pose" and src[1] in _P11:
            xy = pose[:, _P11[src[1]]]
        elif src[0] == "mid" and src[1] in _P11 and src[2] in _P11:
            xy = (pose[:, _P11[src[1]]] + pose[:, _P11[src[2]]]) / 2.0
        elif src[0] == "mid":
            # The 2D contract intentionally drops the hips.  Put the root a
            # stable shoulder-width-scaled distance below the shoulders so
            # the torso remains drawable without inventing hand motion.
            sh = (pose[:, _P11[11]] + pose[:, _P11[12]]) / 2.0
            xy = sh + np.array([0.0, 0.45])
        elif src[0] == "hand":
            xy = hands[src[1]][:, src[2]]
        else:
            continue
        P[:, i, :2] = xy
        P[:, i, 2] = 0.0
    return P, meta


def positions(npz, hybrid=True, hand_z_scale=1.0):
    """-> (T, J, 3) joint positions, NaN where a hand was not detected.

    With `hybrid` (the default) the in-plane coordinates come from the 2D
    landmarks and only depth comes from the world landmarks.

    The world landmarks are the wrong source for anything the viewer can see
    head-on. Measured over 799 hands, finger extension (fingertip to wrist over
    palm length) reads 1.87 for the index in 2D and 1.32 in 3D; anatomy puts an
    extended finger near 1.85, so the 2D figure is right and the 3D one is a
    third short. Worse, the spread between clips collapses -- thumb extension
    varies with std 0.41 in 2D and 0.12 in 3D -- so every handshape is pulled
    towards the same half-curled default. Handshape is a phonological parameter
    in Auslan, so a rig driven by world landmarks signs the wrong word. The
    same bias raises the arm too little: a wrist sits about 0.11 shoulder
    widths lower relative to the shoulder line in 3D than in 2D, a constant
    offset that is not perspective (the ratio rises rather than falls with
    height, r=+0.26, the opposite of what foreshortening would do).

    Depth still has to come from the world landmarks -- 2D has none -- and it
    is what fixes palm orientation and which hand is in front. Mixing a
    perspective source with an orthographic one is not a consistent camera
    model; it is a deliberate trade for a rig that is only ever viewed head-on.
    """
    d = np.load(npz, allow_pickle=False)
    if "pose_world" not in d.files:
        return _positions_2d(d)
    meta = json.loads(str(d["meta"]))
    pose = d["pose_world"].astype(np.float64)
    hands = {"L": d["left_hand_world"].astype(np.float64),
             "R": d["right_hand_world"].astype(np.float64)}
    T = pose.shape[0]
    P = np.full((T, len(SKEL), 3), np.nan)
    for i, (_, _, src, _, _) in enumerate(SKEL):
        if src[0] == "pose":
            P[:, i] = pose[:, src[1]]
        elif src[0] == "mid":
            P[:, i] = (pose[:, src[1]] + pose[:, src[2]]) / 2.0
        else:
            P[:, i] = hands[src[1]][:, src[2]]
    if not hybrid:
        return P, meta

    w, h = float(meta.get("width", 0)), float(meta.get("height", 0))
    if not (w and h):
        return P, meta
    p2 = d["pose"][:, :, :2].astype(np.float64).copy()
    p2[:, :, 0] *= w / h                       # undo the anisotropic squash
    h2 = {"L": d["left_hand"][:, :, :2].astype(np.float64).copy(),
          "R": d["right_hand"][:, :, :2].astype(np.float64).copy()}
    # MediaPipe hand-world landmarks are expressed in a hand-local frame,
    # while the pose-world wrist is in the body frame.  Using the hand-world
    # z values directly makes every finger aim from a different origin than
    # its wrist; the resulting palm normal and finger roll can flip even when
    # the 2D hand is correct.  Keep the measured hand depth, but translate it
    # so landmark 0 (the wrist) is anchored to the pose wrist.
    hand_z = {}
    for side, pose_wrist in (("L", 15), ("R", 16)):
        hw = hands[side]
        rel_z = hw[:, :, 2] - hw[:, 0:1, 2]
        hand_z[side] = pose[:, pose_wrist, 2:3] + rel_z
    for k in h2:
        h2[k][:, :, 0] *= w / h

    sl2, sr2 = p2[:, _P11[11]], p2[:, _P11[12]]
    sw2 = np.nanmedian(np.linalg.norm(sl2 - sr2, axis=-1))
    sw3 = np.nanmedian(np.linalg.norm(pose[:, 11] - pose[:, 12], axis=-1))
    if not (np.isfinite(sw2) and sw2 > 1e-6 and np.isfinite(sw3)):
        return P, meta
    # One scale for the clip, matching how the rest of the pipeline normalises:
    # a per-frame scale would also cancel the signer leaning in and out.
    k = sw3 / sw2
    a2 = (sl2 + sr2) / 2.0                                  # 2D shoulder mid
    a3 = (pose[:, 11, :2] + pose[:, 12, :2]) / 2.0          # world shoulder mid

    def place(i, xy):
        ok = np.isfinite(xy[:, 0]) & np.isfinite(a2[:, 0])
        P[ok, i, :2] = (xy[ok] - a2[ok]) * k + a3[ok]

    for i, (_, _, src, _, _) in enumerate(SKEL):
        if src[0] == "pose" and src[1] in _P11:
            place(i, p2[:, _P11[src[1]]])
        elif src[0] == "mid" and src[1] in _P11 and src[2] in _P11:
            place(i, (p2[:, _P11[src[1]]] + p2[:, _P11[src[2]]]) / 2.0)
        elif src[0] == "hand":
            place(i, h2[src[1]][:, src[2]])
            ok = np.isfinite(hand_z[src[1]][:, src[2], 0])
            P[ok, i, 2] = hand_z[src[1]][ok, src[2], 0]
        # hips have no 2D counterpart (the subset stops at the shoulders), so
        # they keep their world values -- they are stable and never in shot.

    # Depth flattening is available but OFF by default, after being tried and
    # withdrawn.
    #
    # The world hand really is too thick -- 9.9 x 12.0 x 6.2 cm against a real
    # open hand's 9 x 18 x 3 -- and scaling depth by 0.5 does bring thickness
    # to a plausible 3.1 cm. It was introduced on the theory that excess depth
    # was what made fingers project short and read as curled. Measured, it
    # changed projected finger extension from 1.58 to 1.53: no effect.
    #
    # It does have an effect elsewhere, and a bad one. Flattening the hand
    # makes the palm more parallel to the image plane, which forces the palm
    # NORMAL to point along the depth axis: over 120 clips the median |n_z|
    # rises from 0.61 to 0.84. Palm orientation is a phonological parameter in
    # Auslan, and this renders a palm-down hand as one facing the camera.
    # A change that fixed nothing and distorted that is not worth keeping.
    if hand_z_scale != 1.0:
        for side in ("L", "R"):
            w = IDX[f"wrist_{side}"]
            idx = [i for i in range(len(SKEL))
                   if NAMES[i].endswith(f"_{side}") and SKEL[i][2][0] == "hand"]
            P[:, idx, 2] = (P[:, w, 2][:, None] +
                            (P[:, idx, 2] - P[:, w, 2][:, None]) * hand_z_scale)
    return P, meta


def _unit(v, eps=1e-9):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def reference(rule, P):
    """Secondary axis per frame, (T,3). Anatomy supplies these, not the bone."""
    if rule == "hipaxis":
        # The hip landmarks are the root's own position source, so the pelvis
        # axis is not available as a joint; the shoulder axis is the stable
        # stand-in and keeps the root's roll defined.
        return P[:, IDX["shoulder_R"]] - P[:, IDX["shoulder_L"]]
    if rule == "shoulderaxis":
        return P[:, IDX["shoulder_R"]] - P[:, IDX["shoulder_L"]]
    if rule.startswith("armplane"):
        s = rule[-1]
        a = P[:, IDX[f"elbow_{s}"]] - P[:, IDX[f"shoulder_{s}"]]
        b = P[:, IDX[f"wrist_{s}"]] - P[:, IDX[f"elbow_{s}"]]
        n = np.cross(a, b)
        # A straight arm has no plane; fall back to the shoulder axis so the
        # roll stays defined instead of spinning arbitrarily.
        bad = np.linalg.norm(n, axis=-1) < 1e-4
        n[bad] = (P[:, IDX["shoulder_R"]] - P[:, IDX["shoulder_L"]])[bad]
        return n
    if rule.startswith("palm"):
        s = rule[-1]
        return _palm_normal(P, s)
    raise ValueError(rule)


def _palm_normal(P, side):
    """Palm normal by fitting a plane to the whole palm, not two edges of it.

    The obvious construction -- cross(index_mcp - wrist, pinky_mcp - wrist) --
    collapses exactly when the hand does. In a fist those two knuckles sit
    close together, the cross product shrinks towards zero, and its direction
    is then set by landmark noise. Because that normal is what fixes the
    wrist's roll, the result is a wrist that spins while the real hand is
    simply opening and closing, which is what WHALE showed: the sign raises a
    fist and opens it, and the rig turned the wrist instead.

    Fitting a plane through the five palm points (wrist plus the four finger
    MCPs) by SVD uses every one of them, so no single pair going degenerate
    steers the answer, and the smallest singular direction is the normal.
    Continuity is enforced afterwards: the plane fit gives a normal up to
    sign, and an unresolved flip would read as the hand snapping over.
    """
    idx = [IDX[f"wrist_{side}"]] + [IDX[f"{f}1_{side}"]
                                    for f in ("index", "middle", "ring", "pinky")]
    Q = P[:, idx]                                        # (T,5,3)
    T = Q.shape[0]
    out = np.full((T, 3), np.nan)
    prev = None
    for t in range(T):
        pts = Q[t][np.isfinite(Q[t]).all(axis=1)]
        if pts.shape[0] < 3:
            out[t] = prev if prev is not None else np.nan
            continue
        c = pts.mean(axis=0)
        try:
            n = np.linalg.svd(pts - c)[2][-1]
        except np.linalg.LinAlgError:
            out[t] = prev if prev is not None else np.nan
            continue
        if prev is not None and float(n @ prev) < 0:
            n = -n
        out[t] = prev = n
    if prev is None:
        return np.zeros((T, 3))
    # Frames before the first successful fit inherit the first good normal.
    first = np.flatnonzero(np.isfinite(out[:, 0]))
    if len(first):
        out[:first[0]] = out[first[0]]
    return out


def world_frames(P):
    """-> (T, J, 3, 3) world rotation per joint, NaN-propagating."""
    T = P.shape[0]
    R = np.tile(np.eye(3), (T, len(SKEL), 1, 1))
    for i in range(len(SKEL)):
        if AIM[i] < 0:
            continue
        x = _unit(P[:, AIM[i]] - P[:, i])
        r = _unit(reference(REF[i], P))
        z = np.cross(x, r)
        nz = np.linalg.norm(z, axis=-1, keepdims=True)
        # Bone parallel to the reference: pick any perpendicular rather than
        # emit a NaN frame that would poison the whole chain.
        alt = np.tile(np.array([0.0, 0.0, 1.0]), (T, 1))
        z = np.where(nz < 1e-6, np.cross(x, alt), z)
        z = _unit(z)
        y = np.cross(z, x)
        R[:, i] = np.stack([x, y, z], axis=-1)
    return R


def rest_from(P, R):
    """Canonical offsets: each joint's position in its parent's frame."""
    off = np.zeros((len(SKEL), 3))
    for i in range(len(SKEL)):
        p = PARENT[i]
        if p < 0:
            continue
        v = np.einsum("tji,tj->ti", R[:, p], P[:, i] - P[:, p])
        off[i] = np.nanmedian(v, axis=0)
    return off


def locals_from(R):
    """World frames -> parent-relative rotations, as (T, J) Rotation arrays."""
    T = R.shape[0]
    q = np.zeros((T, len(SKEL), 4))
    for i in range(len(SKEL)):
        p = PARENT[i]
        M = R[:, i] if p < 0 else np.einsum("tji,tjk->tik", R[:, p], R[:, i])
        ok = np.isfinite(M).all(axis=(1, 2))
        q[ok, i] = Rot.from_matrix(M[ok]).as_quat()
        q[~ok, i] = np.nan
    return q


# Per-frame provenance of a joint's rotation. A teaching app must be able to
# tell these apart: an invented handshape is worse than an absent one, because
# an absent hand is honestly missing while an invented one teaches a wrong sign.
MEASURED, INTERPOLATED, HELD, ABSENT = 0, 1, 2, 3


# Finger joints. Their absences are filled by HOLDING, never by interpolating.
def _finger_joints():
    return [i for i in range(len(SKEL))
            if SKEL[i][2][0] == "hand"
            and not NAMES[i][-3] == "4"]     # tips carry no rotation anyway


FINGER_JOINTS = frozenset(_finger_joints())


def fill_gaps(q, max_interp_frames=10, policy="honest", max_edge_hold=4):
    """Classify and (where defensible) fill missing rotations.

    Absence has three causes and they do not deserve the same treatment:

        occluded briefly   for the arm, the hand was there before and after
                           and the pose between is a short interpolation --
                           invented, but bounded, and flagged INTERPOLATED.
                           For the FINGERS it is held instead. An arm position
                           is a trajectory and interpolating along it estimates
                           where the wrist really was; a handshape is a
                           configuration, and interpolating between two of them
                           travels through shapes that are neither. Measured on
                           BALD, whose hand is occluded mid-sign, the shapes
                           either side of the gap differ by 22 degrees, so the
                           filled frames showed the hand morphing through
                           postures the signer never made -- in a language where
                           handshape distinguishes words.
        out of shot, or
        never detected     there is nothing to interpolate between. Under
                           policy="honest" these frames stay ABSENT and the
                           renderer must hide the hand rather than show a
                           fabrication -- except within `max_edge_hold` frames
                           of the clip's own start or end, where the nearest
                           measured pose is held instead: a hand that vanishes
                           for the final frame alone is a blink, not honesty
        resting, unused    a one-handed sign leaves the other hand still;
                           holding the last measured pose is correct here, and
                           is what HELD means

    An earlier version filled every hole the same way, and gave a hand that was
    never detected at all an identity rotation -- a straight default hand,
    present in frame, doing something the signer never did. That is the one
    outcome to avoid.

    policy="hold" restores the old always-continuous behaviour for callers that
    need a rig with no gaps and will not present it as ground truth.

    Returns (q, status) with status in {MEASURED, INTERPOLATED, HELD, ABSENT}.
    """
    T, J, _ = q.shape
    status = np.full((T, J), MEASURED, np.int8)
    for j in range(J):
        ok = np.isfinite(q[:, j, 0])
        if ok.all():
            continue
        if not ok.any():
            # Never detected in this clip. There is no pose to hold and none to
            # interpolate; anything we write here is fiction.
            q[:, j] = np.array([0.0, 0.0, 0.0, 1.0])
            status[:, j] = HELD if policy == "hold" else ABSENT
            continue
        idx = np.flatnonzero(ok)
        sl = Slerp(idx, Rot.from_quat(q[idx, j]))
        for m in np.flatnonzero(~ok):
            before = idx[idx < m]
            after = idx[idx > m]
            bridgeable = (len(before) and len(after)
                          and after[0] - before[-1] <= max_interp_frames)
            if bridgeable and j in FINGER_JOINTS:
                q[m, j] = q[before[-1], j]
                status[m, j] = HELD
            elif bridgeable:
                q[m, j] = sl([m]).as_quat()[0]
                status[m, j] = INTERPOLATED
            elif len(before):
                # Past the last sighting. A short tail is held: freezing the
                # last measured pose is not inventing one, and marking a single
                # trailing frame ABSENT makes both hands vanish for one frame,
                # which in a looping clip reads as a blink at the seam. SPIRIT
                # LEVEL lost its hands on frame 69 of 70 exactly this way.
                q[m, j] = q[before[-1], j]
                status[m, j] = (HELD if policy == "hold"
                                or m - before[-1] <= max_edge_hold else ABSENT)
            else:
                # Before the first sighting, by the same argument.
                q[m, j] = q[after[0], j]
                status[m, j] = (HELD if policy == "hold"
                                or after[0] - m <= max_edge_hold else ABSENT)
    return q, status


def uncertain_mask(status, min_run=3):
    """Which joint-frames should be DRAWN as uncertain, as opposed to merely
    being uncertain.

    Fading an interpolated or held joint tells the viewer not to trust it. But
    a single frame of fade among measured ones is a blink: it communicates
    nothing and reads as the rig glitching. POLICE has exactly one held frame
    in ninety-four, and that one frame was the flicker. Only runs long enough
    to be seen as a state -- a few frames, not one -- earn the fade.

    Absence is deliberately NOT included: an absent joint is not drawn at all,
    at any run length, because the alternative is inventing a handshape.
    """
    st = np.asarray(status)
    soft = (st == INTERPOLATED) | (st == HELD)
    out = np.zeros_like(soft)
    T = soft.shape[0]
    for j in range(soft.shape[1]):
        t = 0
        while t < T:
            if not soft[t, j]:
                t += 1
                continue
            u = t
            while u < T and soft[u, j]:
                u += 1
            if u - t >= min_run:
                out[t:u, j] = True
            t = u
    return out


def fk(q, rest, root):
    """Local rotations + rest offsets -> world positions and frames."""
    T = q.shape[0]
    R = np.zeros((T, len(SKEL), 3, 3))
    P = np.zeros((T, len(SKEL), 3))
    for i in range(len(SKEL)):
        Ri = Rot.from_quat(q[:, i]).as_matrix()
        p = PARENT[i]
        if p < 0:
            R[:, i] = Ri
            P[:, i] = root
        else:
            R[:, i] = np.einsum("tij,tjk->tik", R[:, p], Ri)
            P[:, i] = P[:, p] + np.einsum("tij,j->ti", R[:, p], rest[i])
    return P, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="/tmp/retarget")
    ap.add_argument("--world-only", action="store_true",
                    help="drive everything from world landmarks (the old "
                         "behaviour); see positions() for why that flattens "
                         "handshape")
    ap.add_argument("--policy", choices=["honest", "hold"], default="honest",
                    help="honest: leave unbridgeable gaps ABSENT so the "
                         "renderer hides the hand; hold: always emit a pose")
    ap.add_argument("--no-smooth", action="store_true",
                    help="skip the medoid+One Euro filter")
    ap.add_argument("--rest-from", default=None,
                    help="npz whose median proportions define the rig; "
                         "default is the clip itself")
    a = ap.parse_args()

    P, meta = positions(a.npz, hybrid=not a.world_only)
    R = world_frames(P)
    own_rest = rest_from(P, R)
    rig_rest = own_rest if a.rest_from is None else rest_from(*(
        lambda Q: (Q, world_frames(Q)))(positions(a.rest_from,
                                                 hybrid=not a.world_only)[0]))

    q = locals_from(R)
    q, status = fill_gaps(q, policy=a.policy)
    root = np.nan_to_num(P[:, 0], nan=0.0)

    if not a.no_smooth:
        import smooth as SM
        fps = float(meta.get("fps") or 25.0)
        # Medoid first, One Euro second: the medoid removes the single-frame
        # impulses that would otherwise be smeared across neighbours by any
        # exponential stage, and One Euro then settles the holds.
        q = SM.one_euro(SM.rot_medoid(SM.align_signs(q), 3), fps)

    P_solve, _ = fk(q.copy(), own_rest, root)     # signer's own proportions
    P_rig, _ = fk(q.copy(), rig_rest, root)       # the rig's proportions

    seen = np.isfinite(P).all(axis=-1)
    e_solve = np.linalg.norm(P_solve - P, axis=-1)
    e_rig = np.linalg.norm(P_rig - P, axis=-1)
    print(f"{a.npz}  {P.shape[0]} frames  hands L{meta['left_hand_rate']:.2f} "
          f"R{meta['right_hand_rate']:.2f}")
    for tag, code in (("measured", MEASURED), ("interpolated", INTERPOLATED),
                      ("held", HELD), ("absent", ABSENT)):
        n = int((status == code).sum())
        if n:
            print(f"  {tag:13s} {n:7d} joint-frames "
                  f"({n / status.size * 100:5.1f}%)")
    for tag, e in (("solve (own lengths)", e_solve), ("retarget (rig lengths)", e_rig)):
        v = e[seen]
        print(f"  {tag:24s} median {np.median(v)*1000:6.2f} mm   "
              f"p95 {np.percentile(v,95)*1000:6.2f} mm   max {v.max()*1000:6.2f} mm")
    per = np.where(seen, e_solve, np.nan)
    worst = np.argsort(-np.nan_to_num(np.nanmedian(per, axis=0), nan=-1))[:8]
    print("  worst joints (solve, median mm):",
          ", ".join(f"{NAMES[i]} {np.nanmedian(per[:, i])*1000:.1f}" for i in worst))
    # How rigid is the measurement in the first place? A fixed skeleton cannot
    # reproduce a bone whose measured length changes frame to frame.
    L = []
    for i in range(len(SKEL)):
        p = PARENT[i]
        if p < 0: continue
        d = np.linalg.norm(P[:, i] - P[:, p], axis=-1)
        d = d[np.isfinite(d)]
        if len(d) > 5: L.append((NAMES[i], np.median(d), d.std()))
    rel = np.array([s_ / m for _, m, s_ in L if m > 1e-4])
    print(f"  bone-length jitter: median {np.median(rel)*100:.1f}% of length, "
          f"p95 {np.percentile(rel,95)*100:.1f}%")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "fps": meta.get("fps"),
        "joints": NAMES,
        "parents": PARENT,
        "restOffsets": rig_rest.round(6).tolist(),
        "rootPositions": root.round(6).tolist(),
        "rotations": np.nan_to_num(q, nan=0.0).round(6).tolist(),  # xyzw, local
        "status": status.tolist(),   # 0 measured 1 interp 2 held 3 absent
        "statusLegend": {"0": "measured", "1": "interpolated",
                         "2": "held", "3": "absent"},
    }, open(f"{out}.json", "w"))
    np.savez(f"{out}.npz", P_src=P, P_rig=P_rig, q=q, rest=rig_rest,
             root=root, status=status)
    print(f"  -> {out}.json  {out}.npz")


if __name__ == "__main__":
    main()
