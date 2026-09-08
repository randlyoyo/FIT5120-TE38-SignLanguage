# Research notes: the 3D ablation, and where framing guidance came from

Two questions were settled here on 2026-09-08. Both produced results that
argue against the obvious course of action, which is why they are written
down rather than just acted on.

The scripts are from the training workspace, not this repository — they read
`keypoints/`, `keypoints3d/` and `cache/`, none of which are versioned here.
They are included so the numbers below can be reproduced and audited, not so
they can be run from a clone.

---

## 1. Metric 3D keypoints do not help. They cost accuracy where it matters.

**Motivation.** `Test_MTV`, the multi-view split, scored far below every
frontal split. The diagnosis on record was viewpoint: shoulder foreshortening
runs 0.73–3.10 there against 1.68–1.82 everywhere else, and no image-plane
transform can undo an off-axis projection. MediaPipe's world landmarks are
metric, so a clip can be rotated to a canonical facing before comparison —
the one principled fix available.

**Method.** `dtw_features3d.py` builds a 146-dim feature (48 points × 3 + 2
presence flags) in a per-clip torso frame: up from hip midpoint to shoulder
midpoint, right from the shoulder vector orthogonalised against it, forward
from their cross product. Every other stage — fps resample, shoulder-width
scale, motion trim, presence flags — is the 2D builder's, unchanged, so a
difference is attributable to the canonicalisation and the z channel rather
than to a hundred small pipeline changes. Hyper-parameters were copied from
the shipped 2D run verbatim; re-tuning would have made a win unattributable.

**Result.** 3D loses on four of six splits.

| split | both-hands detection | 2D AUC | 3D AUC | 2D rank-1 | 3D rank-1 | Δ |
|---|---|---|---|---|---|---|
| Valid | 0.681 | 0.9994 | 0.9992 | 85.91% | 86.03% | +0.12 |
| Test_STU | 0.787 | 0.9996 | 0.9997 | 89.47% | 89.92% | +0.45 |
| Test_ITW | 0.735 | 0.9993 | 0.9992 | 88.47% | 85.00% | **−3.47** |
| Test_TED | 0.635 | 0.9990 | 0.9985 | 81.91% | 81.02% | −0.89 |
| Test_SYN | 0.676 | 0.9967 | 0.9955 | 81.76% | 78.85% | −2.91 |
| Test_MTV | 0.397 | 0.8940 | 0.9056 | 22.89% | 25.93% | **+3.04** |

Both large effects are real, and they point in opposite directions
(`mcnemar.py`, paired over identical query sets):

    Test_MTV   +3.04pt   chi2 59.1   p = 1.5e-14
    Test_ITW   −3.47pt   chi2 63.8   p = 1.4e-15

**Why it fails.** The viewpoint diagnosis was half right. Off-axis capture
does break recognition, but not through projection geometry — through
self-occlusion. `Test_MTV` has both hands present in only 0.397 of frames
against 0.64–0.79 on frontal splits. Canonical rotation can straighten the
landmarks you have; it cannot produce a hand that was never detected. And on
frontal capture, where x/y already carry the available information, the extra
z channel is largely driven by MediaPipe's body prior rather than by image
evidence, so it dilutes rather than informs. This is the same conclusion an
earlier handshape-clustering attempt reached independently (3D 0.166 overlap
against 2D 0.196).

**Decision.** Keep the 2D encoder. `Test_MTV` was never the deployment
scenario — the product is a single frontal camera — so the split 3D helps on
is the one that does not matter, and the split it hurts on, `Test_ITW`, is
the closest thing in the corpus to a real user.

One observation worth keeping: on MTV the two models agree far less than
expected (both correct 1849, 3D-only 1475, 2D-only 1085). Their union ceiling
is 34.4%. Still unusable, so this is a note for the report, not a design.

## 2. Framing guidance (`website/client/src/lib/framing.ts`)

Measured over 3,600 clips from Train + Test_ITW + Test_STU (p2–p98):

    shoulder span   0.177 – 0.257   of frame height, aspect-corrected
    shoulder mid x  0.491 – 0.525
    shoulder mid y  0.463 – 0.539
    head/shoulder   0.357 – 0.423   frontal; 0.249 – 0.502 on MTV

Findings that shaped the rules:

- **Distance does not need to be matched.** The feature pipeline divides by
  shoulder width, so scale is normalised out of the model's input. What
  breaks at close range is that the signing space stops fitting: hands reach
  1.40 shoulder widths above the shoulder midpoint (p99), so a span above
  ~0.35 pushes a raised hand off the top edge. At 0.6 m from a typical webcam
  the span is 0.69 — signing into a laptop at desk distance cannot work, at
  any model quality. Users must be told to step back to roughly 1.5–2 m.
- **Head-to-shoulder ratio cannot measure distance.** It is a body constant
  (0.39 at every distance). It measures turn, and does so well: frontal
  splits sit in 0.357–0.423, the multi-view split spreads to 0.249–0.502.
- **The far limit is hand pixels, not span.** The whole corpus is 512×408
  with hands measuring 42–47 px, and even there both hands are found in only
  0.63–0.79 of frames. Below ~37 px the rate falls to 0.64. Because pixels
  depend on stream resolution, the far limit must too: at 640×480 a user is
  too far at 1.8 m, at 1280×720 not until 2.4 m. Requesting 720p is the
  cheapest single improvement to the capture path and is exported as
  `VIDEO_CONSTRAINTS`.

**Consequence for the capture UI.** A user standing 1.5–2 m away cannot reach
the keyboard, and stopping late is expensive — an earlier measurement put
rank-1 at 33.1% when recording continued for 100% of the sign's duration past
the end, against 86.0% for a sign performed three times too slowly. Capture
should therefore start automatically once framing holds (`makeStabiliser`)
and stop automatically on stillness, rather than on a button.

## Files

    dtw_features3d.py   3D feature builder, torso-frame canonicalisation
    mkcache3d.py        feature cache builder for the 3D pipeline
    eval_all.py         score one encoder on every split, tau fixed on Valid
    mcnemar.py          paired significance test; takes a split name
    framing_probe.py    apparent size vs detection rate vs accuracy
    run_c.sh            the 3D training run, hyper-parameters copied from 2D
    status_c.sh         progress view for that run
    results/            the two result files the tables above come from

The trained 3D encoder is deliberately not committed. It is a negative
result and must not reach deployment.
