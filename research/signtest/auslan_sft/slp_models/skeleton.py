"""Joint layout of the pose representation. The single source of truth.

The layout is a superset of the Uni-Sign pose-only input (unisign/spec.py,
verified against Uni-Sign commit eed438b), so that the frozen, pretrained
Uni-Sign pose encoder can read generated poses without any re-mapping:

    positions  0..8    body      COCO-WholeBody [0, 3..10]
    positions  9..29   left hand COCO-WholeBody 91..111   (signer's own left)
    positions 30..50   right hand COCO-WholeBody 112..132 (signer's own right)
    positions 51..68   face_all  jaw [23,25,..,39] + inner lip 83..90 + nose tip 53
    positions 69..78   eyebrows  COCO-WholeBody 40..49    (NOT seen by Uni-Sign)

The eyebrows are extra: Auslan marks questions, topicalisation and negation on
the brows, so the generator should produce them even though the pretrained
encoder never modelled them.

"left"/"right" are the signer's anatomical sides as COCO-WholeBody defines
them. For a frontal signer the anatomical left hand appears on the image's
right. Nothing in this project mirrors frames; handedness is linguistic.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

LAYOUT_VERSION = "auslan-sft-wholebody79-v1"

BODY_WB = [0, 3, 4, 5, 6, 7, 8, 9, 10]
LHAND_WB = list(range(91, 112))
RHAND_WB = list(range(112, 133))
FACE_WB = list(range(23, 40))[::2] + list(range(83, 91)) + [53]
BROW_WB = list(range(40, 50))

WHOLEBODY_INDICES: np.ndarray = np.asarray(
    BODY_WB + LHAND_WB + RHAND_WB + FACE_WB + BROW_WB, dtype=np.int64)
NUM_JOINTS = len(WHOLEBODY_INDICES)            # 79
COORD_DIM = 2                                   # rtmlib gives 2D image keypoints

PART_SLICES: dict[str, slice] = {
    "body": slice(0, 9),
    "left": slice(9, 30),
    "right": slice(30, 51),
    "face_all": slice(51, 69),
    "brows": slice(69, 79),
}
# Parts consumed by the Uni-Sign pose encoder, in the order models.py iterates.
UNISIGN_PARTS = ("body", "left", "right", "face_all")

# Named positions inside the 79-joint layout.
NOSE = 0
L_EAR, R_EAR = 1, 2
L_SHOULDER, R_SHOULDER = 3, 4
L_ELBOW, R_ELBOW = 5, 6
L_WRIST_BODY, R_WRIST_BODY = 7, 8
L_HAND_ROOT, R_HAND_ROOT = 9, 30
FACE_NOSE_TIP = 68

# Hand topology, identical to Uni-Sign stgcn_layers/gcn_utils.py layout 'left'.
_HAND_EDGES_LOCAL = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # little
]
FINGER_OF_EDGE = [e // 4 for e in range(20)]  # 0 thumb .. 4 little

BODY_EDGES = [
    (NOSE, L_EAR), (NOSE, R_EAR),
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST_BODY),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST_BODY),
]
LHAND_EDGES = [(a + L_HAND_ROOT, b + L_HAND_ROOT) for a, b in _HAND_EDGES_LOCAL]
RHAND_EDGES = [(a + R_HAND_ROOT, b + R_HAND_ROOT) for a, b in _HAND_EDGES_LOCAL]
# Body wrist and hand-model wrist are separate detections of the same joint.
WRIST_LINKS = [(L_WRIST_BODY, L_HAND_ROOT), (R_WRIST_BODY, R_HAND_ROOT)]

_F = PART_SLICES["face_all"].start
FACE_EDGES = ([(_F + i, _F + i + 1) for i in range(8)]                 # jaw
              + [(_F + 9 + i, _F + 10 + i) for i in range(7)]          # inner lip
              + [(_F + 16, _F + 9)])                                   # close lip
_B = PART_SLICES["brows"].start
BROW_EDGES = ([(_B + i, _B + i + 1) for i in range(4)]
              + [(_B + 5 + i, _B + 6 + i) for i in range(4)])

ALL_EDGES = BODY_EDGES + WRIST_LINKS + LHAND_EDGES + RHAND_EDGES + FACE_EDGES + BROW_EDGES
# Bones whose length is anatomically stable enough to constrain (arms, fingers).
BONE_LOSS_EDGES = BODY_EDGES[2:] + LHAND_EDGES + RHAND_EDGES


def part_weights(hand: float = 1.0, body: float = 1.0, face: float = 1.0,
                 brows: float = 1.0) -> np.ndarray:
    w = np.ones(NUM_JOINTS, dtype=np.float32)
    w[PART_SLICES["body"]] = body
    w[PART_SLICES["left"]] = hand
    w[PART_SLICES["right"]] = hand
    w[PART_SLICES["face_all"]] = face
    w[PART_SLICES["brows"]] = brows
    return w


def layout_metadata() -> dict:
    meta = {
        "layout_version": LAYOUT_VERSION,
        "num_joints": NUM_JOINTS,
        "coord_dim": COORD_DIM,
        "wholebody_indices": WHOLEBODY_INDICES.tolist(),
        "part_slices": {k: [v.start, v.stop] for k, v in PART_SLICES.items()},
    }
    blob = json.dumps(meta, sort_keys=True).encode()
    meta["fingerprint"] = hashlib.sha256(blob).hexdigest()[:16]
    return meta


def _self_check() -> None:
    assert NUM_JOINTS == 79
    assert WHOLEBODY_INDICES[FACE_NOSE_TIP] == 53, "nose tip must end face_all"
    assert WHOLEBODY_INDICES[L_HAND_ROOT] == 91 and WHOLEBODY_INDICES[R_HAND_ROOT] == 112
    assert WHOLEBODY_INDICES[L_SHOULDER] == 5 and WHOLEBODY_INDICES[R_SHOULDER] == 6
    for a, b in ALL_EDGES:
        assert 0 <= a < NUM_JOINTS and 0 <= b < NUM_JOINTS


_self_check()
