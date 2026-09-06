# Sign Verification — interface specification

Scope: **isolated single words, real-time webcam capture, 2D keypoints, frontal
camera.** The learner picks a word, signs it, and the system answers *does this
match the stored templates of that word*. Continuous signing, sentences, and
off-axis cameras are out of scope — see [Limits](#limits).

Everything below is a contract. The encoder was trained on features produced by
one exact pipeline; a client that deviates anywhere will still run, still return
plausible-looking numbers, and be silently wrong. [Conformance](#conformance)
exists for that reason.

---

## 1. Overview

```
webcam frame ──▶ MediaPipe Holistic ──▶ 48 landmarks/frame
                                            │
                                            ▼
                                    feature pipeline  (§3)
                                            │
                                            ▼  (T=96, 98) float32
                                    encoder.onnx  (§4)
                                            │
                                            ▼  256-d unit vector
                        cosine distance vs the word's 12 templates  (§5)
                                            │
                                            ▼
                                    matched: distance < τ
```

All inference runs **in the browser**. Video never leaves the device; the only
network call fetches 12 KB of template vectors for the selected word.

---

## 2. Landmark extraction

**Model:** MediaPipe Holistic Landmarker, bundle `float16/1`
(`models/holistic_landmarker.task` — ship this file, do not re-download).

**JS package:** `@mediapipe/tasks-vision@0.10.33`. The version must match the
`mediapipe==0.10.33` used to extract the training data. A different bundle or a
different major/minor shifts the landmark distribution and costs accuracy in a
way that looks like a model problem, not a version problem.

**Running mode:** `VIDEO` (`detectForVideo`), not `IMAGE`. VIDEO carries
tracking state across frames; IMAGE gives jumpier coordinates and breaks
train/inference consistency. Timestamps must be strictly increasing integers in
milliseconds.

**No mirroring.** Feed frames exactly as the camera decodes them. If the preview
is mirrored for the user (common and good UX), mirror the *display* only, never
the tensor. A mirrored tensor swaps left and right hands and the model will
score a correct sign as wrong.

### Landmarks actually consumed

Only 48 of the landmarks the model produces are used. Face is **not** used at
all — it may be requested or not, it makes no difference.

| group | source | count | order |
|---|---|---|---|
| arms | `poseLandmarks` indices `[11, 12, 13, 14, 15, 16]` | 6 | left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist |
| left hand | `leftHandLandmarks` `0..20` | 21 | MediaPipe hand order |
| right hand | `rightHandLandmarks` `0..20` | 21 | MediaPipe hand order |

Only `x` and `y` are read. `z` is discarded — it is a relative estimate on three
different origins (pose to the hips, hands to the wrist) and is not comparable
across groups.

A group that was not detected in a frame is **missing as a whole** — MediaPipe
returns all 21 hand points or none. Represent a missing group as `null`, and
never as zeros: `(0,0)` is the top-left corner, a legal coordinate the model
would read as a real hand position.

### Per-frame record

```ts
interface Frame {
  poseArms: [number, number][] | null;   // 6 points, normalised [0,1]
  leftHand: [number, number][] | null;   // 21 points
  rightHand: [number, number][] | null;  // 21 points
}

interface Capture {
  frames: Frame[];
  width: number;    // video element videoWidth
  height: number;   // video element videoHeight
  fps: number;      // measured capture rate, see §3.2
}
```

---

## 3. Feature pipeline

Reference implementation: `signtest/dtw_features.py`, function `build()`. The
steps below must be applied **in this order** — resampling before trimming,
centring before scaling.

### 3.1 Aspect-ratio correction

```
x ← x * (width / height)
```

MediaPipe normalises `x` by frame width and `y` by frame height independently,
so a physically square gesture arrives stretched by the frame's aspect ratio.
The training data was uniformly 512×408; a phone in portrait is 0.56. The
shoulder-width divisor applied later is a single scalar and cannot undo an
anisotropic squash. Skipping this step cost measurable accuracy on mixed-aspect
data.

### 3.2 Temporal resample to 25 fps

```
duration = n_frames / fps
n_out    = rint(duration * 25)
out[j]   = in[ clamp(rint((j + 0.5) / 25 * fps - 0.5), 0, n_frames - 1) ]
```

**`rint` is round-half-to-EVEN, not round-half-up.** This is not a detail. The
training features were produced by `numpy.round`, which breaks ties to even;
JavaScript's `Math.round` breaks them upward. Going from 30 fps to 25, the
expression `(j + 0.5) * 30/25 - 0.5` lands on exactly `x.5` every fifth frame,
so the two rules pick a different source frame for **20% of all frames**. It
does not throw, it does not look wrong, and it quietly costs accuracy.

```js
function rint(v) {
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
```

Use the same `rint` for the trim padding in §3.5.

**Nearest neighbour, not interpolation.** Hand landmarks are all-or-nothing per
frame; interpolating across a gap invents coordinates that were never observed.

This maps frames onto real time. It does **not** equalise duration — a slow
signer still produces more frames than a fast one. That is deliberate: measured
robustness to uniform speed change is total (2× slower and 3× slower both score
within 0.1% of the original).

Reject the capture if `fps < 5` or `fps > 120`. Browser capture rate varies;
measure it from actual frame timestamps rather than trusting a nominal 30.

### 3.3 Centre on the shoulder midpoint

```
origin = (leftShoulder + rightShoulder) / 2      // per frame
```

Subtract from every point. Removes where the person stands in frame. For frames
where pose was not detected, carry the last known origin forward.

### 3.4 Scale by shoulder width

```
scale = median over frames of |leftShoulder - rightShoulder|
```

Divide every point by this single scalar. **One median for the whole clip, not
per frame** — a per-frame divisor would also normalise away the signer leaning
in and out, which is real motion. Reject if `scale < 1e-4`.

### 3.5 Trim the still lead-in and lead-out

```
speed[t]  = max over both wrists of |p[t+1] - p[t]|
threshold = 0.15 * percentile(speed, 95)   // linear interpolation between
                                           // order statistics (numpy default)
active    = indices where speed > threshold
pad       = rint(0.1 * 25)                   // 3 frames ≈ 100 ms
start     = max(0, active.first - pad)
stop      = min(T, active.last + 2 + pad)
```

Clips open and close with the hands at rest. Those frames carry no sign, differ
in length between captures, and dilute the resample. Measured on the training
data: hand detection runs 0.83 across the middle 60% of a clip against 0.54 over
the outer 40%.

Reject if fewer than 8 frames survive.

### 3.6 Assemble the 98-dim frame vector

```
[0..11]   arms      6 points × (x, y)     order as in §2
[12..53]  left hand  21 points × (x, y)
[54..95]  right hand 21 points × (x, y)
[96]      leftHandPresent   1.0 or 0.0
[97]      rightHandPresent  1.0 or 0.0
```

Missing coordinates become `0.0` **after** the presence flag is set. The flag is
what carries "this hand is not here" — zero-filling alone would make a missing
hand nearly free, and one-handed versus two-handed is a strong discriminative
cue (6.1% of training clips are effectively one-handed).

### 3.7 Resample to fixed length T = 96

Linear interpolation along the time axis to exactly 96 frames:

```
idx = linspace(0, n - 1, 96)
out[j] = in[floor(idx[j])] * (1 - frac) + in[min(floor(idx[j]) + 1, n - 1)] * frac
```

Linear here, unlike §3.2 — by this point the presence flags have been set and a
fractional flag is a meaningful "the hand was present for part of this window".

Result: `Float32Array(96 * 98)`, row-major, no NaN.

---

## 4. Encoder

**File:** `models/encoder.onnx` (3.7 MB)
**Runtime:** `onnxruntime-web` ≥ 1.20

| | name | shape | dtype |
|---|---|---|---|
| input | `x` | `[batch, 96, 98]` | float32 |
| output | `emb` | `[batch, 256]` | float32 |

The output is already L2-normalised — do not normalise again.

Measured 2.12 ms per single-item inference on CPU (Node, x86). Batch axis is
dynamic but a real-time client only ever sends 1.

Parity with the PyTorch source is 2.4e-07 max absolute difference.

---

## 5. Decision rule

```
distance = 1 - max over i of dot(embedding, template[i])     // i = 0..11
matched  = distance < tau
```

`tau = 0.5294347405433655`, from `models/bank_index.json`.

The templates are unit vectors and so is the embedding, so the dot product is
cosine similarity directly.

**Minimum over templates, not mean.** Verification asks whether the attempt
matches *any* stored example of the word; averaging lets one atypical template
drag a good match down.

### Threshold provenance

`tau` is the equal-error-rate point measured on the Valid split: **EER 0.73%,
AUC 0.9994**. It was fixed on Valid and applied unchanged to the test splits —
never re-tuned per split.

**Treat it as a starting value, not a constant.** It was measured on
studio-recorded, green-screen, Kinect footage. Real webcam captures in real
rooms will differ. Expose it as configuration and re-calibrate once real user
data exists. See [Limits](#limits).

---

## 6. Template API

The client needs only the selected word's templates — 12 KB, not the whole
39.5 MB bank.

### `GET /api/recognition/template?word=<GLOSS>`

```json
{
  "word": "WHALE",
  "dim": 256,
  "n": 12,
  "tau": 0.5294347405433655,
  "vectors": [[0.031, -0.118, ...], ...]
}
```

`404` if the gloss is not in the 3215-word vocabulary.

**Server implementation.** `models/bank.f32` is a flat float32 array of shape
`(3215, 12, 256)` in C order. `models/bank_index.json` gives the word list; a
word's offset is `wordIndex * 12288` bytes, length `12288`.

```js
const idx = index.words.indexOf(word);          // build a Map at startup
const off = idx * index.bytes_per_word;
const buf = bank.subarray(off, off + index.bytes_per_word);
```

Loading `bank.f32` once into memory at startup (39.5 MB) is the simplest option
and well within a Railway dyno.

### Optional: `GET /api/recognition/vocabulary`

Returns the 3215 glosses, so the UI can show which library signs are
verifiable.

---

## 7. Conformance

`test/golden.json` holds three real captures with their expected outputs. A
client implementation **must** pass these before being trusted with user data.

Each case gives:

- `input` — `width`, `height`, `fps`, and raw `pose` / `left_hand` /
  `right_hand` arrays exactly as MediaPipe would hand them over. `-999` marks a
  missing landmark (JSON has no NaN).
- `expect.feat_len` — frame count after §3.1–3.6
- `expect.feat_first_frame`, `expect.feat_last_frame` — the 98-dim vectors
- `expect.fixed_checksum` — sum of the whole `(96, 98)` tensor
- `expect.embedding` — the 256-d output

Tolerance **1e-5** on every value. A correct implementation lands near 3e-7 —
if you are sitting at 1e-5 exactly, something is subtly off rather than merely
imprecise.

The fixture values are exact float32 promoted to float64. Do not round them
when reformatting: at 6 decimal places the input rounding alone propagates to
~1e-5 after the shoulder-width division, which swallows the whole tolerance
budget.

`scripts/verifyDoc.mjs` is a worked implementation of §3 written from this
document alone, and it passes. Read it if a step here is ambiguous — but treat
this document as the contract and that file as one reading of it.

Note the `input.pose` arrays carry all 11 landmarks of the stored subset
`[0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]`; the arms are entries `5..10` of that
list. A live client reading `poseLandmarks` directly takes indices
`[11, 12, 13, 14, 15, 16]` of the full 33-point pose, which is the same six
points.

This test is the only thing standing between a correct port and a subtly wrong
one. A feature pipeline that is off by a factor, an axis, or an ordering does
not throw — it degrades.

---

## 8. Capture guidance for the UI

Derived from measured failure modes, not guesswork.

**Uniform speed does not matter.** A learner signing 3× slower than the
reference scores identically (85.9% → 86.0% top-1). Do not rush them.

**Pauses mid-sign do matter.** Inserting still frames equal to 150% of the
original duration drops top-1 from 85.9% to 46.2%. Restarting mid-capture
(a false start followed by the real attempt) drops it to 76.9%.

So the UI should:

- give a clear "go" cue and a short countdown, so the attempt starts cleanly
- keep the capture window tight (roughly 2–5 s) rather than open-ended
- offer an obvious retry rather than letting a learner correct mid-capture
- reject and ask for a retake when fewer than 8 frames survive trimming, or
  when neither hand was detected in over 60% of frames

**Frontal camera.** Accuracy falls off steeply with camera angle — see below.

---

## 9. Limits

Known, measured, and unresolved. None of these are bugs.

**Closed vocabulary — 3215 words.** There is no "I don't know" output. A sign
outside the vocabulary returns the nearest of the 3215 with confidence. For
verification this is harmless (the learner chose the word), but do not reuse the
same call for open-ended identification without adding a rejection rule.

**Isolated words only.** Training clips are one word, 2–4 s. A sentence will be
forced onto a single word.

**Frontal only.** Measured on the multi-view split, binned by how square-on the
signer is (shoulder width ÷ head-to-shoulder distance; frontal training data
sits at 1.68–1.82):

| frontality | top-1 |
|---|---|
| 0.00–0.90 (strong profile) | 0.8% |
| 1.20–1.45 | 18.4% |
| 1.60–1.90 (matches training) | 36.7% |

**Domain gap is larger than the viewpoint gap.** Note the last row: even at
training-matched frontality, the multi-view split reaches only 36.7% against
85.9% on the studio split. Controlling for hand-detection quality as well, the
best-conditioned multi-view clips still reach only ~52%. The residual is real
rooms, consumer cameras, and unseen signers — the exact conditions this app will
run in.

**So the headline numbers are an upper bound.** EER 0.73% and top-1 85.9% were
measured on green-screen Kinect footage. Real webcam performance is somewhere
between that and the ~50% seen on consumer-camera captures, and nobody has
measured where. **Measuring it is the point of this first integration.** Until
that number exists:

- keep `tau` configurable, do not hard-code it in client logic
- run internally before exposing to learners — a false "you got it wrong" is
  costly feedback to give a beginner
- log the raw distance alongside every decision so the threshold can be
  re-derived from real captures rather than re-guessed

**Beginner-specific behaviour is unmeasured.** The pause figures in §8 come from
synthetic distortions applied to fluent recordings, not from real beginners.
Real hesitation may look different — a hand hovering rather than freezing, for
instance, which the §3.5 motion threshold would not catch.

---

## 10. Files

```
recognition/
├── API.md                      this document
├── models/
│   ├── encoder.onnx            3.7 MB   ship to the browser
│   ├── holistic_landmarker.task 13 MB   ship to the browser
│   ├── manifest.json                    vocabulary, dims, tau
│   ├── bank.f32                39.5 MB  server-side only
│   └── bank_index.json                  word → offset
└── test/
    └── golden.json             0.6 MB   conformance fixtures
```

Provenance of the encoder: `signtest/runs/final/` — config, training history and
per-split results are in `result.json`; the 24-trial hyper-parameter search that
selected it is in `signtest/runs/sweep/trials.jsonl`.
