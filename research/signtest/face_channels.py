#!/usr/bin/env python3
"""Per-frame face and head channels, as blendshape weights and bone angles.

    python face_channels.py keypoints3d/Valid/79612_kf_rgb.npz

These are the channels an avatar needs that the skeleton does not carry. They
are NOT decoration: in Auslan a raised brow marks a yes/no question, a lowered
brow a wh-question, and head turn marks role shift, so an idle animation must
never write to them. Blinking is the opposite case -- the corpus has no eyelid
landmarks at all, so a blink cannot be measured and must be generated.

Two things make this work despite the face being stored in 2D only:

* A blendshape weight is a scalar, not a position, so the depth that is missing
  from the 18 face points is not needed to drive one.
* Head rotation does not come from the face points. Nose, eyes and ears are in
  the 33-point pose, which IS stored in metric 3D, so a full head frame is
  available even though MediaPipe emits no world landmarks for the face.

Brow height is normalised against the signer's own median. Measured over 300
clips, brow height varies more BETWEEN people (std 0.103 of interpupillary
distance) than within a clip (0.042), so an un-centred channel would render a
naturally high-browed signer as permanently asking a question.
"""
import argparse
import json

import numpy as np

import slr_common as C

P = {v: i for i, v in enumerate(C.POSE_SUBSET)}
# MediaPipe pose: 0 nose, 2 LEFT eye, 5 RIGHT eye, 7 LEFT ear, 8 RIGHT ear.
# (slr_common.py's comment on this line has left and right the wrong way round;
# it is only a comment, but it is what made the first roll calculation wrap.)
NOSE, EYE_L, EYE_R, EAR_L, EAR_R = 0, 2, 5, 7, 8
CHANNELS = ["browRaise", "browAsym", "mouthOpen", "mouthWide",
            "headYaw", "headPitch", "headRoll"]

# Landmark geometry, not posture: the ear markers sit slightly above and behind
# the nose, so the ear-to-nose vector is not level even on a level head. Taken
# as the corpus-wide median over 400 clips, where the between-clip spread is
# only ~4 degrees, so this is a fixed property of the landmarks rather than
# something that varies with the signer. Subtracting a constant rather than
# each clip's own median keeps a signer who genuinely holds their head low.
YAW_OFFSET_DEG = -5.03
PITCH_OFFSET_DEG = -12.48


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def extract(npz_path, smooth=True):
    d = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    w, h = float(meta.get("width", 0)), float(meta.get("height", 0))
    face = d["face"][:, :, :2].astype(np.float64).copy()
    pose2 = d["pose"][:, :, :2].astype(np.float64).copy()
    if w > 0 and h > 0:                      # undo the anisotropic squash
        face[:, :, 0] *= w / h
        pose2[:, :, 0] *= w / h
    pw = d["pose_world"].astype(np.float64) if "pose_world" in d.files else None
    T = pose2.shape[0]

    # --- scale: interpupillary distance, the only face-local ruler available
    ipd = np.linalg.norm(pose2[:, P[EYE_L]] - pose2[:, P[EYE_R]], axis=-1)
    s = float(np.nanmedian(ipd))
    if not np.isfinite(s) or s < 1e-5:
        raise ValueError("no usable interpupillary distance")

    lips, brows = face[:, :C.N_LIPS], face[:, C.N_LIPS:]
    eye_y = (pose2[:, P[EYE_L], 1] + pose2[:, P[EYE_R], 1]) / 2

    # Brow height above the eye line. Sign is flipped so that positive means
    # RAISED, matching the blendshape's own direction.
    brow = (eye_y[:, None] - brows[:, :, 1]) / s
    brow_raise = np.nanmean(brow, axis=1)
    brow_asym = np.nanmean(brow[:, 3:], axis=1) - np.nanmean(brow[:, :3], axis=1)

    mouth_open = (np.nanmax(lips[:, :, 1], axis=1) -
                  np.nanmin(lips[:, :, 1], axis=1)) / s
    mouth_wide = (np.nanmax(lips[:, :, 0], axis=1) -
                  np.nanmin(lips[:, :, 0], axis=1)) / s

    # --- head frame from the pose's own 3D points when available
    #
    # Forward must come from the EARS, not the eyes. The nose sits below the
    # eye line as well as in front of it, so nose-minus-eye-centre points
    # mostly downwards and reads a level head as pitched down 76 degrees. The
    # ear midpoint is behind the nose and level with it, so nose-minus-ear
    # -centre is forward with only the real pitch left in it.
    if pw is not None:
        lateral = _unit(pw[:, EAR_L] - pw[:, EAR_R])  # towards left
        fwd = pw[:, NOSE] - (pw[:, EAR_L] + pw[:, EAR_R]) / 2
        fwd = _unit(fwd - lateral * np.sum(fwd * lateral, axis=-1, keepdims=True))
        right = lateral
        yaw = np.degrees(np.arctan2(fwd[:, 0], -fwd[:, 2]))
        pitch = np.degrees(np.arcsin(np.clip(-fwd[:, 1], -1, 1)))
        roll = np.degrees(np.arctan2(-right[:, 1], right[:, 0]))
    else:
        # The 2D extractor has no reliable depth axis.  Keep yaw/pitch neutral
        # rather than inventing a turn from foreshortened image coordinates;
        # roll is still observable from the eye line.
        right = pose2[:, P[EYE_R]] - pose2[:, P[EYE_L]]
        yaw = np.zeros(T)
        pitch = np.zeros(T)
        roll = np.degrees(np.arctan2(-right[:, 1], right[:, 0]))

    out = np.stack([brow_raise, brow_asym, mouth_open, mouth_wide,
                    yaw - YAW_OFFSET_DEG, pitch - PITCH_OFFSET_DEG, roll],
                   axis=1)

    # Centre the per-signer channels on this clip's own resting value. Head
    # angles keep their absolute zero: "facing the camera" is meaningful, while
    # "this signer's neutral brow" is not comparable between people.
    for i in (0, 1, 2, 3):
        out[:, i] -= np.nanmedian(out[:, i])
    out[:, 6] -= np.nanmedian(out[:, 6])       # camera roll is not articulation

    out = np.where(np.isfinite(out), out, 0.0)
    if smooth:
        out = _lowpass(out, float(meta.get("fps") or 25.0))
    return out, meta


def _lowpass(x, fps, cutoff=3.0):
    """One-pole low-pass. These channels are ratios of noisy 2D landmarks and
    are read as continuous weights, so a little lag costs nothing here -- unlike
    the hand rotations, where lag is the first thing a viewer notices."""
    a = 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) * fps)
    y = x.copy()
    for t in range(1, len(x)):
        y[t] = y[t - 1] + a * (x[t] - y[t - 1])
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ch, meta = extract(a.npz)
    print(f"{a.npz}  {ch.shape[0]} frames")
    print(f"{'channel':>11s}{'median':>9s}{'p5':>8s}{'p95':>8s}{'range':>8s}")
    for i, n in enumerate(CHANNELS):
        c = ch[:, i]
        print(f"{n:>11s}{np.median(c):9.2f}{np.percentile(c,5):8.2f}"
              f"{np.percentile(c,95):8.2f}{c.max()-c.min():8.2f}")
    if a.out:
        json.dump({"channels": CHANNELS, "fps": meta.get("fps"),
                   "values": ch.round(4).tolist()}, open(a.out, "w"))
        print("->", a.out)


if __name__ == "__main__":
    main()
