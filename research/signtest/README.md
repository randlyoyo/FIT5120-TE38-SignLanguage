# Auslan keypoint extraction — verification pipeline

Three scripts plus a shared config. Run them in order. Do not skip to training
until the checks pass.

```
slr_common.py   config: which landmarks, which connections, quality thresholds
extract.py      videos -> .npz keypoint files
overlay.py      .npz + source video -> annotated mp4 (layer 1: eyeball it)
stats.py        .npz directory -> quality report + CSV (layer 2: numbers)
```

## Setup

```bash
pip install "mediapipe==0.10.33" opencv-python numpy
# Pin it. mediapipe 1.0.x aborts on macOS arm64 inside the holistic graph
# ("graph_service.h:139 Check failed: service_"), and the version must match
# slr_common.MEDIAPIPE_VERSION anyway.
```

Then download the Holistic Landmarker model bundle. Get the exact URL from the
official page rather than trusting a link pasted in a chat — Google moves these:

<https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker>

Save it as `holistic_landmarker.task` next to the scripts. **Record which
version you downloaded in `slr_common.MODEL_TAG`.** If you later re-download a
different bundle, your old keypoints are no longer comparable to your new ones.

## Workflow

```bash
# 1. extract  (start with 20 words, NOT the full dataset)
python extract.py --videos ./clips --out ./keypoints --model ./holistic_landmarker.task

# 2. numbers
python stats.py --npz-dir ./keypoints --csv ./checks/quality.csv

# 3. eyes — watch the clips stats.py flagged as worst
python overlay.py --videos ./clips --npz-dir ./keypoints --out-dir ./checks --limit 20
```

## The mirror test — do this before anything else

Record yourself raising **only your right hand**. Run it through `extract.py`
and `overlay.py`. In the output video the **orange** skeleton must be on the
hand you actually raised.

If it is blue, something in your chain is mirroring the frame. `extract.py`
does no mirroring at all — frames go in as decoded. The bug will be either in
how you recorded the clip or, later, in your browser code.

Repeat this test **in the browser** once your JS path exists. Browsers usually
mirror the webcam preview so it looks like a mirror to the user. If you mirror
the displayed canvas but feed unmirrored frames to MediaPipe (or the reverse),
left and right swap. Nothing errors. Accuracy just quietly collapses.

Auslan distinguishes dominant and non-dominant hand, so a left/right swap is a
different sign, not a cosmetic issue.

## What the flags mean

| flag | meaning | what to do |
|---|---|---|
| `low_dominant_hand` | the busier hand was missing in too many frames | watch it; if the signer's hand leaves frame, drop the clip |
| `long_both_hand_gap` | both hands lost for a long stretch | usually occlusion or motion blur — likely unusable |
| `low_pose` | body not detected reliably | framing or lighting problem |
| `frame_count_mismatch` | decoded frame count ≠ container metadata | codec issue; check the clip opens correctly |
| `frozen_frames` | identical consecutive poses | tracker stalled or the video has duplicate frames |
| `out_of_bounds` | coordinates far outside [0,1] | signer partly out of frame |
| `no_motion` | nothing moved | wrong clip, or extraction silently failed |
| `all_nan` | nothing detected at all | extraction failed |
| consistency block at the end | more than one schema/mediapipe version in the set | re-extract everything with one pinned config |

`low_dominant_hand` firing on a handful of clips is normal. Firing on 30% of
your set means your extraction settings are wrong, not that your data is bad.

## Important: what these scripts do NOT prove

They prove the extraction ran without crashing and produced plausible numbers.
They say nothing about whether the keypoints carry enough information to tell
Auslan signs apart.

For that you need the layer-4 test: take 20 words with multiple signers each,
and check that the within-word distance is clearly smaller than the
between-word distance — or just train a small classifier and see if it beats
5% (random over 20 classes) by a wide margin. Half a day of work, and it is the
only result that actually justifies scaling up.

## Design decisions baked in

* **VIDEO running mode, not IMAGE.** VIDEO carries tracking state between
  frames and is what the browser will use. Changing this changes your numbers.
* **Missing landmarks are `NaN`, never `0.0`.** Zero is a valid coordinate
  (top-left corner); a model trained on zero-filled gaps learns that a missing
  hand is a hand resting in the corner.
* **Landmark selection follows the Kaggle Google-ISLR convention** (2023
  "Top-1st to 5th Solutions" report), 71 points per frame:

  | group | points | note |
  |---|---|---|
  | hands | 21 + 21 = 42 | kept whole |
  | pose | 11 | nose, eyes, ears, shoulders, elbows, wrists |
  | face | 18 | outer lip contour 12 + eyebrows 6 |

  In 2D that is **142 dims/frame**. Dropping face entirely — a common strong
  baseline — leaves 53 points / 106 dims; do that in the dataloader, not by
  re-extracting.
* **Everything below the hips is discarded.** Sign meaning lives in the upper-body
  signing space; lower-body points are negative noise.
* **No normalisation at extraction time.** Raw normalised image coordinates are
  stored. Shoulder-centring and shoulder-width scaling belong in your training
  dataloader, where you can change them without re-extracting 282K clips.
