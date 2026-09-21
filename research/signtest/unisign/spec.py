#!/usr/bin/env python3
"""Pose representation spec — the single source of truth for the whole pipeline.

VERIFIED against the Uni-Sign repository (commit eed438b), not transcribed from
the paper. The authorities are:

    datasets.py :: load_part_kp        index selection and normalisation
    datasets.py :: crop_scale          the scale every part is divided by
    stgcn_layers/gcn_utils.py :: Graph joint ORDER, via the graph topology
    models.py   :: Uni_Sign.__init__   3 input channels, part names
    demo/pose_extraction.py            what the stored keypoints actually are

Read that list before changing anything here. The indices must match
pre-training exactly; a mismatch does not raise, it just makes the weights
meaningless and shows up weeks later as bad BLEU.

Four facts that are easy to get wrong, and were:

  1. Joint ORDER is load-bearing, not just the set of indices. Each part is an
     ST-GCN over a fixed graph, so joint k is wired to specific neighbours and
     specific weights. The face group is jaw(9), inner lip(8), nose tip(1) --
     the nose is LAST, it is the graph centre, and it is the root everything
     else is measured from. Putting it anywhere else silently scrambles the
     face branch.

  2. Input is THREE channels per joint: x, y, confidence. `proj_linear` is
     nn.Linear(3, 64). Confidence is a model input, not a loading detail.

  3. Normalisation is not "subtract a root". A single scale is computed from
     the BODY's bounding box over the whole clip (crop_scale); the body is
     mapped into [-1,1] by it, and the hands and face are root-subtracted and
     then divided by that same body scale. The parts are therefore on one
     common scale, which is what makes hand size relative to torso meaningful.

  4. Left and right hands SHARE an encoder. models.py does
     `gcn_modules['left'] = gcn_modules['right']`, likewise the projection.
     There are four branches but three sets of weights. Any plan that says the
     four per-part encoders are independent is describing a different model.
"""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

# Bump whenever anything here changes the meaning of a stored/loaded tensor.
SPEC_VERSION = "unisign-pose-v2-verified"

NUM_WHOLEBODY_KEYPOINTS = 133

# Confidence at or below this is treated as absent: the joint is zeroed in all
# three channels. datasets.py :: load_part_kp, thr = 0.3.
CONF_THRESHOLD = 0.3

# Part names as the model expects them. Order matters: models.py iterates
# self.modes and concatenates the features in this order before pose_proj.
PART_ORDER: tuple[str, ...] = ("body", "left", "right", "face_all")

# --- Index selection, 0-based, copied from load_part_kp ---------------------
#   body      skeleton[[0] + list(range(3, 11))]
#   left      skeleton[91:112]
#   right     skeleton[112:133]
#   face_all  skeleton[list(range(23, 40))[::2] + list(range(83, 91)) + [53]]

PART_INDICES: dict[str, np.ndarray] = {
    "body": np.asarray([0] + list(range(3, 11)), dtype=np.int64),
    "left": np.arange(91, 112, dtype=np.int64),
    "right": np.arange(112, 133, dtype=np.int64),
    "face_all": np.asarray(
        list(range(23, 23 + 17))[::2] + list(range(83, 83 + 8)) + [53],
        dtype=np.int64,
    ),
}

# Root joint, as a position within the part. Hands use their wrist (element 0),
# the face uses the nose tip (the LAST element). The body has no root: it is
# normalised by crop_scale instead.
PART_ROOT_LOCAL: dict[str, int | None] = {
    "body": None, "left": 0, "right": 0, "face_all": -1,
}

PART_SIZES: dict[str, int] = {p: len(PART_INDICES[p]) for p in PART_ORDER}

# Branches that share one set of weights (models.py). Recorded here because the
# freezing policy and any parameter accounting has to know.
SHARED_ENCODERS: tuple[tuple[str, str], ...] = (("left", "right"),)

# Face landmarks NOT selected, and so not representable at all.
#
#   modelled:     mouthing (inner lip) and head orientation (jaw contour).
#                 Mouthing matters here: Auslan is BANZSL, shares BSL's heavy
#                 English mouthing, and many signs are distinguished by mouth
#                 pattern alone. That channel is present.
#
#   NOT modelled: eyebrows (68-point face indices 41-50, i.e. wholebody 63-72
#                 0-based) and eyes (60-71 -> 82-93). Auslan marks polar
#                 questions, wh-questions, topicalisation, conditionals and
#                 negation with brow position, and uses gaze for pronominal
#                 reference. None of it reaches the encoder.
#
# Adding them changes the face branch's node count, which changes its graph and
# discards its pre-trained weights. That is a separate experiment, not a config
# change.
FACE_UNMODELLED = {
    "eyebrows": list(range(23 + 17, 23 + 27)),
    "eyes": list(range(23 + 36, 23 + 48)),
}


def _self_check() -> None:
    expect = {"body": 9, "left": 21, "right": 21, "face_all": 18}
    for part, n in expect.items():
        if PART_SIZES[part] != n:
            raise AssertionError(f"{part}: expected {n} joints, got {PART_SIZES[part]}")
    # The face graph's centre is the last node and it must be the nose tip
    # (wholebody index 53). gcn_utils.Graph('face_all') sets center = 17.
    if int(PART_INDICES["face_all"][-1]) != 53:
        raise AssertionError("face_all: nose tip (53) must be the LAST joint")
    if int(PART_INDICES["left"][0]) != 91 or int(PART_INDICES["right"][0]) != 112:
        raise AssertionError("hand roots must be the wrist, at position 0")
    for part, idx in PART_INDICES.items():
        if idx.min() < 0 or idx.max() >= NUM_WHOLEBODY_KEYPOINTS:
            raise AssertionError(f"{part}: index outside COCO-WholeBody range")


_self_check()


def schema_fingerprint() -> str:
    """Hash over everything defining the representation, stored in every clip."""
    payload = {
        "spec_version": SPEC_VERSION,
        "part_order": list(PART_ORDER),
        "part_indices": {p: PART_INDICES[p].tolist() for p in PART_ORDER},
        "part_root_local": PART_ROOT_LOCAL,
        "conf_threshold": CONF_THRESHOLD,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


SCHEMA_FINGERPRINT = schema_fingerprint()


# --- Normalisation: a port of datasets.py, kept deliberately close to it ----

def crop_scale(motion: np.ndarray, thr: float = CONF_THRESHOLD):
    """Port of datasets.py::crop_scale. motion is (T, J, 3) = x, y, conf.

    Maps into [-1, 1] using a single bounding box over every confident joint in
    the whole clip -- not per frame. A per-frame box would rescale the signer
    every time a hand extends, and the motion would be normalised away.
    """
    result = copy.deepcopy(motion)
    valid = motion[motion[..., 2] > thr][:, :2]
    if len(valid) < 4:
        return np.zeros(motion.shape, dtype=np.float32), 0.0, None
    xmin, xmax = float(valid[:, 0].min()), float(valid[:, 0].max())
    ymin, ymax = float(valid[:, 1].min()), float(valid[:, 1].max())
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape, dtype=np.float32), 0.0, None
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result.astype(np.float32), scale, [xs, ys]


def load_part_kp(keypoints: np.ndarray, scores: np.ndarray,
                 thr: float = CONF_THRESHOLD) -> dict[str, np.ndarray]:
    """Port of datasets.py::load_part_kp.

    keypoints: (T, 133, 2), ALREADY divided by [W, H] at extraction time.
    scores:    (T, 133)
    returns:   {part: (T, J, 3)} float32, x/y/confidence, ready for the model.

    The body is normalised first because everything else is divided by the
    scale it produces. If the body is unusable (fewer than four confident
    joints in the entire clip) the scale is 0 and every part comes back zeroed
    -- the clip carries no usable geometry and quietly rescaling it would
    invent one.
    """
    out: dict[str, np.ndarray] = {}
    scale: float | None = None

    for part in PART_ORDER:
        idx = PART_INDICES[part]
        kps = keypoints[:, idx, :].astype(np.float64, copy=True)
        conf = scores[:, idx].astype(np.float64, copy=True)

        root = PART_ROOT_LOCAL[part]
        if root is not None:
            # Per frame, subtract the part's own root joint. Hands use the
            # wrist (0), the face uses the nose tip (-1, the last joint).
            kps = kps - kps[:, [root], :]

        stacked = np.concatenate([kps, conf[..., None]], axis=-1)

        if part == "body":
            result, scale, _ = crop_scale(stacked, thr)
        else:
            assert scale is not None, "body must be normalised first"
            if scale == 0:
                result = np.zeros(stacked.shape, dtype=np.float32)
            else:
                result = stacked.copy()
                result[..., :2] = result[..., :2] / scale
                result = np.clip(result, -1, 1)
                result[result[..., 2] <= thr] = 0
                result = result.astype(np.float32)
        out[part] = result.astype(np.float32)
    return out


def describe() -> str:
    lines = [
        f"spec={SPEC_VERSION} fingerprint={SCHEMA_FINGERPRINT}",
        f"channels per joint: 3 (x, y, confidence)   conf threshold: {CONF_THRESHOLD}",
        f"shared encoders: {SHARED_ENCODERS}",
    ]
    for part in PART_ORDER:
        root = PART_ROOT_LOCAL[part]
        idx0 = PART_INDICES[part].tolist()
        root_txt = ("crop_scale, no root" if root is None
                    else f"local {root} = wholebody {idx0[root]} (0-based)")
        lines.append(f"  {part:<9} n={PART_SIZES[part]:>3}  root={root_txt}")
        lines.append(f"    0-based wholebody indices: {idx0}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
