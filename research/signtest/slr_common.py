"""Shared constants for the Auslan keypoint pipeline.

IMPORTANT: every value in this file is part of your data contract.
If you change anything here after extracting your training set, you MUST
re-extract. Bump SCHEMA_VERSION whenever you touch it.
"""

SCHEMA_VERSION = "2.0.0"   # 2.0.0: switched to Kaggle GISLR landmark selection

# Pin these. Do NOT use @latest anywhere -- if Google ships a new model your
# training data and your browser inference silently drift apart.
MODEL_FILENAME = "holistic_landmarker.task"
MODEL_TAG = "float16/1"          # record which bundle you downloaded
MEDIAPIPE_VERSION = "0.10.33"    # must match the JS @mediapipe/tasks-vision major/minor

# ---------------------------------------------------------------------------
# Landmark counts produced by MediaPipe Holistic Landmarker
# ---------------------------------------------------------------------------
N_POSE = 33
N_HAND = 21
N_FACE_FULL = 478

# ---------------------------------------------------------------------------
# Landmark selection follows the Kaggle Google-ISLR competition convention
# (see the 2023 "Isolated Sign Language Recognition: Top-1st to 5th Solutions"
# report). Hands are kept whole; pose is cut to the signing space; the 478-point
# face mesh is cut to the two groups that actually carry non-manual markers.
#
#   hands  21 + 21 = 42
#   pose             11
#   face             18   (outer lip contour 12 + eyebrows 6)
#   ------------------------
#   total            71  points/frame  ->  142 dims/frame in 2D
#
# Dropping face entirely (a common strong baseline) leaves 53 points / 106 dims.
# extract.py stores face regardless; drop it at training time if you want that
# variant -- re-extracting just to change the selection is not worth hours.
# ---------------------------------------------------------------------------

# Outer lip ring, sampled evenly: corners, upper arc, lower arc.
FACE_LIPS = [
    61,                 # left corner (image-right)
    40, 37,             # upper left arc
    0,                  # upper mid (cupid's bow)
    267, 270,           # upper right arc
    291,                # right corner
    321, 314,           # lower right arc
    17,                 # lower mid
    84, 91,             # lower left arc
]

# Eyebrows, 3 per brow: inner / mid / outer.
FACE_BROWS = [
    107, 105, 70,       # right brow (image-left)
    336, 334, 300,      # left brow
]

FACE_SUBSET = FACE_LIPS + FACE_BROWS
N_FACE = len(FACE_SUBSET)
N_LIPS = len(FACE_LIPS)

# ---------------------------------------------------------------------------
# Pose subset -- signing space only: head anchors, shoulders, elbows, wrists.
# Everything from the hips down is dropped on purpose. The literature is
# consistent that lower-body points are pure negative noise for sign meaning,
# and on a seated/webcam framing they are usually out of shot anyway.
# Indices follow MediaPipe Pose's 33-point topology.
# ---------------------------------------------------------------------------
POSE_SUBSET = [
    0,              # nose
    2, 5,           # right eye, left eye
    7, 8,           # ears
    11, 12,         # shoulders
    13, 14,         # elbows
    15, 16,         # wrists
]
N_POSE_SUB = len(POSE_SUBSET)

# Drawing connections, expressed as indices INTO POSE_SUBSET
_p = {v: i for i, v in enumerate(POSE_SUBSET)}
POSE_CONNECTIONS = [
    (_p[11], _p[12]),                       # shoulder line
    (_p[11], _p[13]), (_p[13], _p[15]),     # left arm
    (_p[12], _p[14]), (_p[14], _p[16]),     # right arm
    (_p[0], _p[2]), (_p[0], _p[5]),         # nose -> eyes
    (_p[2], _p[7]), (_p[5], _p[8]),         # eyes -> ears
]

# Outer lip drawn as a closed ring; indices INTO FACE_SUBSET.
LIP_CONNECTIONS = [(i, (i + 1) % N_LIPS) for i in range(N_LIPS)]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm closure
]

# ---------------------------------------------------------------------------
# Quality thresholds used by stats.py. Tune these once you have seen the
# distribution on your own data -- the defaults are a starting guess, not law.
# ---------------------------------------------------------------------------
MIN_DOMINANT_HAND_RATE = 0.70   # fraction of frames with the busier hand detected
MIN_POSE_RATE = 0.95
MIN_FRAMES = 12
