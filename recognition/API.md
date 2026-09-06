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

MediaPipe runs **in the browser** — video never leaves the device. The
landmarks (about 190 KB of JSON for a 4-second capture) are posted to the
server, which runs the feature pipeline, the encoder and the template search.

Server-side rather than in-browser inference, deliberately: §3 is the most
failure-prone part of this system, and keeping one implementation means one
thing to keep in step with the training code. It also lets the model change
without shipping a new client, and identification needs the 39.5 MB template
bank resident anyway.

---

## 2. Landmark extraction

**Model:** MediaPipe Holistic Landmarker, bundle `float16/1`. Load it straight
from Google's CDN at the pinned URL — there is no copy to host or deploy:

```
https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task
```

`sha256 e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8` — this
is the exact bundle the training keypoints were extracted with. Keep the `/1/`
in the path; `latest` would silently move under you.

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

**This trims the two ends only, and only what is still.** It is not a sliding
window and it does not search for the sign inside a longer recording. Two
consequences the client has to handle, because the server cannot:

- **A pause in the middle survives.** Everything between the first and last
  moving frame is kept, however much of it is the learner hesitating.
- **Anything that moves survives.** Lowering the hands, reaching for the mouse,
  scratching, adjusting hair — none of that is still, so none of it is trimmed.

What the trim does absorb is dead time at the ends: the moment between pressing
a button and starting to sign, and the stillness after finishing. That part is
genuinely free.

§8 has the measured cost of both cases and what the UI must do about them.

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
**Runtime:** `onnxruntime-node` ≥ 1.20 (the encoder runs server-side)

| | name | shape | dtype |
|---|---|---|---|
| input | `x` | `[batch, 96, 98]` | float32 |
| output | `emb` | `[batch, 256]` | float32 |

The output is already L2-normalised — do not normalise again.

Measured 2.12 ms per single-item inference on CPU (Node, x86). Batch axis is
dynamic but a real-time client only ever sends 1.

Parity with the PyTorch source is 2.4e-07 max absolute difference.

---

## 5. Two modes

The same embedding answers two different questions, and they are not equally
reliable. Keep them apart in the UI.

| | verify | identify |
|---|---|---|
| question | "is this the word I chose?" | "which word was that?" |
| learner supplies | the target word | nothing |
| compares against | that word's 12 templates | all 3215 words |
| measured | **EER 0.73%** | top-1 82–89%, top-4 93–98% |
| output | boolean | four candidates |

Verify is the reliable mode and should carry any practice or assessment
feature. Identify is a lookup aid: top-1 is wrong roughly one time in eight, so
it returns a shortlist for the learner to choose from rather than an answer.

Four candidates rather than five costs 0.5–1.0 points of coverage (Valid 97.4%
to 96.9%, the worst split 93.9% to 92.9%) and was chosen for the UI.

### Verify

```
distance = 1 - max over i of dot(embedding, template[i])     // i = 0..11
matched  = distance < tau
```

`tau = 0.5294347405433655`. **Minimum over templates, not mean** — verification
asks whether the attempt matches *any* stored example, and averaging lets one
atypical template drag a good match down.

`tau` is the equal-error point on the Valid split (EER 0.73%, AUC 0.9994),
fixed there and applied unchanged to every test split. Treat it as a starting
value: it was measured on green-screen studio footage. See [Limits](#limits).

### Identify

Distance to every word by the same rule, then the five smallest.

Confidence is the **margin** between the best and second-best word, not the
absolute distance:

```
margin    = distance[2nd] - distance[1st]
confident = margin >= 0.04
```

Absolute distance is a poor confidence signal here — it predicts a correct
top-1 at AUC 0.70, while the margin reaches 0.88. Distances vary too much
between words for one global cut to mean anything.

**`confident` is advisory. Never use it to hide the candidate list.** Among
low-margin captures the correct word is still in the top four 79–91% of the
time — suppressing those would throw away mostly-good answers. Use it to soften
how the first candidate is presented, nothing more.

At the 0.04 threshold, measured per split:

| split | top-1 | top-4 | flagged confident | top-1 within those | top-4 among the rest |
|---|---|---|---|---|---|
| Valid | 85.9% | 96.9% | 82.0% | 93.9% | 89.8% |
| Test_STU | 89.5% | 97.8% | 83.7% | 96.4% | 90.9% |
| Test_ITW | 88.5% | 97.5% | 83.6% | 95.4% | 90.6% |
| Test_TED | 81.9% | 94.8% | 77.6% | 91.6% | 86.2% |
| Test_SYN | 81.8% | 92.9% | 79.2% | 91.8% | 79.2% |

The last column is why the list is never hidden: even when the margin says the
model is unsure, the answer is usually still in front of the learner.

---

## 6. HTTP API

Base path `/api/recognize`. Every capture body is the same shape:

```ts
interface Capture {
  width: number;            // video element videoWidth
  height: number;           // videoHeight
  fps: number;              // measured, see §3.2
  pose: number[][][];       // [frame][11 landmarks][x, y]
  left_hand: number[][][];  // [frame][21][x, y]
  right_hand: number[][][]; // [frame][21][x, y]
}
```

A missing landmark is `[-999, -999]`; a missing hand is 21 such entries. Never
send zeros for a missing hand — `(0,0)` is the top-left corner, a legal
position the model reads as a real one.

`pose` carries the 11-point subset `[0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]` of
MediaPipe's 33; only entries 5..10 (shoulders, elbows, wrists) are read, so the
other five may be filled with `[-999, -999]` if that is easier to produce.

### `POST /api/recognize/verify`

```json
{ "word": "WHALE", "capture": { ... } }
```

```json
{
  "word": "WHALE",
  "distance": 0.2289,
  "threshold": 0.5294347405433655,
  "matched": true,
  "frames": 83
}
```

`frames` is the count that survived trimming (§3.5) — useful for telling a
learner their capture was mostly still.

`404 unknown_word` if the gloss is outside the 3215-word vocabulary.

### `POST /api/recognize/identify`

```json
{ "capture": { ... } }
```

```json
{
  "candidates": [
    { "word": "QUEER",  "distance": 0.2289 },
    { "word": "BALLET", "distance": 0.4504 },
    { "word": "INTERVIEW", "distance": 0.4859 },
    { "word": "DECLARE (CRICKET)", "distance": 0.4915 }
  ],
  "margin": 0.2215,
  "confident": true,
  "marginThreshold": 0.04,
  "frames": 83
}
```

Always four candidates, ordered nearest first.

### `GET /api/recognize/vocabulary`

`{ "count": 3215, "words": [...] }` — which glosses can be recognised at all.

### Errors

| status | body | meaning |
|---|---|---|
| 400 | `{ error: "unusable_capture", detail }` | too few frames, no motion, no shoulders, fps out of range |
| 404 | `{ error: "unknown_word", word }` | verify only |
| 503 | `{ error: "model_unavailable" }` | model assets missing from the deployment |

`unusable_capture` is a retake prompt, not an error to log — it fires when the
learner barely moved, or the camera lost them.

### Server implementation

`src/recognition/features.js` (§3), `src/recognition/model.js` (encoder +
bank), `src/routes/recognize.js` (the routes above). The bank loads once at
first request: 39.5 MB resident, scoring all 3215 words is one pass over it,
about 1 ms. Encoder inference is 1.6 ms. No vector index is needed at this
size.

`RECOGNITION_MODELS` overrides where the assets are read from; it defaults to
`recognition/models/`.

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

Derived from measured failure modes, not guesswork. The single most important
requirement is at the end of this section: **the recording must stop by itself
when the learner finishes.**

### Speed is free

A learner signing 3× slower than the reference scores identically — top-1
85.9% against 86.0%. Resampling to a common frame rate (§3.2) makes uniform
tempo irrelevant. Do not rush anyone, and do not add a "sign faster" hint.

### Extra content is not free

Everything measured against the same baseline of 85.9% top-1:

| what the capture contains | top-1 |
|---|---|
| the sign, nothing else | **85.9%** |
| + 30% more unrelated movement | 82.3% |
| + 50% | 67.2% |
| + 100% | **33.1%** |
| + 150% | 28.6% |
| a still pause worth 150% of the sign | 46.2% |
| a false start, then the real attempt | 76.9% |

A learner who finishes in 1.5 s inside a fixed 5-second window has recorded
230% extra. That lands in the worst row of this table, and §3.5 cannot remove
it: lowering the hands is movement, not stillness.

### So the recording must stop itself

Fixed-length capture is not acceptable. Stop as soon as the hands come to rest.
The client already has the landmarks live, so this costs a few lines.

**Work in the §3 normalised frame** — shoulder midpoint at the origin, shoulder
width as the unit, y increasing downward. Raw image coordinates will not do:
the thresholds below are in shoulder widths, which is what makes them
independent of how far the learner sits from the camera.

Measured over 800 validation clips, separating the frames the trim discards
(hands at rest) from the frames it keeps (signing):

| | wrist height | per-frame wrist speed |
|---|---|---|
| at rest | median **1.45**, 10–90% [1.33, 1.60] | median 0.006, 90th 0.018 |
| signing | median 0.50, 10–90% [−0.15, 1.44] | median 0.048, 90th 0.187 |

Wrist height is measured as the *higher* of the two wrists, so one raised hand
is enough to count as still signing.

Neither signal is sufficient alone — at rest height alone, 64% of signing
frames also qualify, because plenty of signs are made low. Combine them, and
require the hands to have been **raised at least once** before rest detection
arms at all, otherwise it fires immediately on the stillness before the attempt
starts:

```js
// all values in the §3 normalised frame
const wristY = Math.min(leftWristY, rightWristY);   // higher of the two
const speed  = Math.max(leftWristSpeed, rightWristSpeed);

if (!started && wristY < 0.8) started = true;       // hands came up
if (started) {
  restFrames = (wristY > 1.2 && speed < 0.03) ? restFrames + 1 : 0;
  if (restFrames >= 8) stop();                      // ~0.32 s at 25 fps
}
```

At these thresholds the rule stops mid-sign on **0.1%** of validation clips.
Loosening the speed cut to 0.05 raises that to 0.6%, and dropping the
consecutive-frame count to 5 raises it to 1.8% — the eight-frame requirement is
what makes the rule safe, not the thresholds themselves.

**What this measurement could not check:** how long the rule takes to fire
*after* a sign genuinely ends. The training clips are cut tight to the sign, so
94% of them simply run out of frames before eight rest frames accumulate. In
real use the learner lowers their hands and leaves them there, so the frames
exist — but the trailing latency is unverified, and the fallback ceiling below
is what covers it being wrong.

Keep a hard ceiling — 5 seconds — as a fallback for when rest detection fails,
not as the normal path.

```
countdown 3-2-1
   │
   ├── hands at rest for ~0.3 s ──▶ stop        (the normal case)
   └── 5 s elapsed ───────────────▶ stop        (fallback only)
```

Stopping early also halves the upload: about 95 KB for a 2-second capture
against 240 KB for five.

### The rest of the UI

- Give a clear "go" cue and a countdown, so the attempt starts cleanly rather
  than the learner thinking and signing at the same time — hesitation costs
  more than slowness.
- Make **retry** prominent. A learner who notices a mistake should start over,
  not correct themselves mid-capture: a false start costs 9 points of top-1,
  and correcting mid-capture costs far more.
- Treat `400 unusable_capture` as a retake prompt, not an error. It fires when
  the learner barely moved or the camera lost them.
- Use the `frames` field in the response: a small number means the capture was
  mostly still, which is worth saying out loud before showing a poor result.

### Frontal camera

Accuracy falls off steeply with camera angle — see [Limits](#limits).

---

## 9. Limits

Known, measured, and unresolved. None of these are bugs.

**Closed vocabulary — 3215 words.** There is no "I don't know" output. A sign
outside the vocabulary comes back as the nearest of the 3215. For verify this is
harmless — the learner chose the word. For identify it means the five candidates
are always four real glosses, even when the input was not a sign at all, so the
UI must let the learner reject all four rather than forcing a pick.

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
├── ASSETS.md                   how to obtain the two uncommitted files
├── models/
│   ├── encoder.onnx            3.7 MB   committed
│   ├── bank.i8                 9.9 MB   committed, server-side
│   ├── manifest.json                    vocabulary, dims, tau
│   └── bank_index.json                  word → offset, dtype
├── scripts/
│   ├── verifyDoc.mjs           independent implementation of §3, passes §7
│   ├── buildBank.mjs           bank.npy -> bank.i8, only needed after retraining
│   └── fetchModel.sh           optional local copy of the MediaPipe bundle
└── test/
    └── golden.json             0.6 MB   conformance fixtures

website/server/
├── src/recognition/features.js §3, the port under test
├── src/recognition/model.js    encoder + bank
├── src/routes/recognize.js     §6
└── test/recognition.test.js    npm run test:recognition
```

Provenance of the encoder: `signtest/runs/final/` — config, training history and
per-split results are in `result.json`; the 24-trial hyper-parameter search that
selected it is in `signtest/runs/sweep/trials.jsonl`.
