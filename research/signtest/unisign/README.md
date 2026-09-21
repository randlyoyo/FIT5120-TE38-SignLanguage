# Auslan → English, Uni-Sign pose-only transfer

Implementation of the transfer plan. Separate from the MediaPipe/DTW pipeline in
the repo root: different estimator, different keypoint selection, and the two
must not share extracted keypoints.

The pose representation and the model interface here are **verified against the
Uni-Sign repository** (commit `eed438b`), not transcribed from the paper. Where
the paper and the shipped code disagree, the code wins — it produced the
weights. The disagreements are listed at the bottom.

```
spec.py            the pose representation. Single source of truth.
manifest.py        both corpora -> one JSONL manifest
auslan_daily.py    an Auslan-Daily release -> the records JSON manifest.py reads
extract_pose.py    manifest -> rtmlib keypoints (.npz), upstream-identical
verify_pose.py     the gate: consistency, quality, mirror test, overlays
dataset.py         manifest + .npz -> Uni-Sign's exact batch format, OOV report
model_adapter.py   loads the checkpoint; parameter grouping for freezing
smoke_model.py     a small real model, for testing the harness only
train.py           Arms A / B / C
stitch.py          Arm D: pseudo-continuous augmentation
evaluate.py        BLEU-1..4, ROUGE-L, BLEURT-20, OOV
configs/           one YAML per arm
```

## Install

```bash
pip install torch numpy opencv-python sacrebleu pyyaml
pip install rtmlib onnxruntime      # the estimator Uni-Sign actually uses
pip install transformers sentencepiece einops   # for the real model
```

**mmpose is not needed.** Uni-Sign's own `demo/pose_extraction.py` uses
`rtmlib`, an ONNX runtime with no mmcv/mmdet/mmpose dependency at all.

## Order of operations

Steps 1–3 are a gate, not a formality. Do not start step 4 until step 3 passes.

### 1. Confirm the checkpoint loads

Already downloaded into this project:

```
Uni-Sign/                                 the repo, commit eed438b
Uni-Sign/pretrained_weight/mt5-base/      mT5 weights + tokenizer
checkpoints/csl_stage1_weight.pth         pose-only pre-trained base
checkpoints/how2sign_pose_only_slt.pth    pose-only, fine-tuned for English SLT
checkpoints/openasl_pose_only_slt.pth     likewise, different ASL corpus
```

```bash
python model_adapter.py --checkpoint ../checkpoints/csl_stage1_weight.pth \
  --repo ../Uni-Sign --mt5-path ../Uni-Sign/pretrained_weight/mt5-base --report
```

Verified: loads with `missing=0 unexpected=0`, 587.7M parameters, nothing
`unmatched`. `pose_encoder` holds `proj_linear` + `gcn_modules` (0.41M), which
is what Arm B stage 1 unfreezes; `temporal` holds `fusion_gcn_modules`,
`part_para` and `pose_proj` (4.94M); `decoder` is mT5 (582M, 99%).

A real batch of extracted Auslan poses has been pushed through the loaded
model: loss computes and `generate` returns fluent English. On the untuned
How2Sign checkpoint that English is wrong — "WHALE" comes back as *"And then
I'm going to put it in the sun."* — which is the plan's premise made visible.
The base model has structure and English, and no Auslan lexicon whatsoever.

Read the group table. `unmatched` must be near zero, and `pose_encoder` must
hold `proj_linear` and `gcn_modules` and nothing else — that is what Arm B
stage 1 unfreezes. If the grouping is wrong, Arm B is not the experiment the
plan describes and nothing will say so.

### 2. Build the manifest

```bash
AD="../Auslan-Daily"
T="$AD/Dataset Split Table/Sign Language Translation"

python auslan_daily.py \
  --split-table "$T/Auslan-Daily_Communication.xlsx" \
  --video-root  "$AD/Auslan-Daily Communication/Signer Only Video Clip/Signer.zip" \
  --subset communication \
  --split-table "$T/Auslan-Daily_News.xlsx" \
  --video-root  "$AD/Auslan-Daily News/Signer Only Video Clip/Signer.zip" \
  --subset news \
  --out ../data/auslan_daily_records.json

python manifest.py --out ../data/manifest.jsonl \
  --mmwl-root ../MM-WLAuslan \
  --ad-annotations ../data/auslan_daily_records.json
```

**Nothing is ever unpacked.** Both corpora are read in place: MM-WLAuslan from
its split archives, Auslan-Daily from `Signer.zip`, via `<archive>::<member>`
references that `extract_pose.py` opens directly. The Communication archive
alone is 16 GB and its deflate ratio is 0.997 — video is already compressed, so
the zip is doing no real work and unpacking would cost the disk twice for
nothing. Indexing all 14,041 members takes about a second (zip central
directory), and extraction from the archive is verified end to end.

The split tables are XLSX with columns `Video_Name`, `Video_Clip_Name`,
`Subtitle`, `Split` (train/dev/test), `Signer_ID` and — News only —
`Signer_Video`. **The clip naming is a trap in both directions.** Signer-Only
files are named `<clip>_signer.mp4`; for News that full name sits in
`Signer_Video`, while Communication has no such column and needs the suffix
appended. Key on `Video_Clip_Name` alone and Communication matches zero files.
`auslan_daily.py` tries the id as given, then with `_signer`, and prints the
tally of which rule matched.

Two release quirks, both handled, neither worth "fixing":

* `video_68_179` really is stored without the `_signer` suffix — matched
  exactly.
* `video_12_62` has no signer crop at all; the archive holds only
  `video_12_62_3_nonsigner.mp4`. It is dropped. Loosening the match to catch it
  would pair a non-signer's video with a signing subtitle.

So Communication resolves 14,040 of 14,041 rows.

Real counts from the release: Communication 12,441 / 800 / 800 and News
9,669 / 700 / 700 (train/dev/test). The plan document says 9,665 for News
train; the table says 9,669.

**MM-WLAuslan camera selection is settled by the data, not by guesswork.**
`Train`, `Valid`, `Test_STU`, `Test_ITW`, `Test_TED` and `Test_SYN` contain only
the `kf` studio frontal camera. `Test_MTV` is the multi-device set (`phone0-11`,
`web0-10`) and the default camera filter excludes it. Frontal-only is 70,730
clips across all splits, 38,580 of them training, 3,215 glosses × 12 signers.

**Auslan-Daily: take the Signer-Only clips.** The release ships Signer-Only
(cropped to the signer by ground-truth bounding box), Multi-Person (uncropped),
Whole Video, and its own Pose Annotation. Clips contain one to ten people and
the paper counts that interference as part of the task; cropping removes it at
the source, which beats asking a pose estimator to pick the signer. The bundled
pose annotation is *not* a shortcut past extraction — it is a different keypoint
layout, and mixing it in would break comparability with MM-WLAuslan.

**Licences differ and the strictest one governs.** Auslan-Daily is CC BY 4.0
(commercial use allowed with attribution), but the **Uni-Sign weights are
CC BY-NC 4.0 — non-commercial**. Anything fine-tuned from them inherits that.
If this project has a commercial destination, that constraint is upstream of
every experiment here and needs settling before, not after.

### 3. Extract, then verify

```bash
# pilot first -- 200 clips, not 70,000
python extract_pose.py --manifest data/manifest.jsonl --out data/pose --limit 200
python verify_pose.py --npz-dir data/pose --csv checks/quality.csv \
                      --exclude-out checks/excluded.txt --overlay-dir checks/overlays
```

Extraction mirrors `demo/pose_extraction.py`: `rtmlib.Wholebody`,
`mode="lightweight"` (the default in both of Uni-Sign's own scripts), and
coordinates divided by `[W, H]`. All 133 keypoints are stored; the selection
happens in `spec.py` at load time, so an index correction never costs a
re-extraction.

`verify_pose.py` exits non-zero if anything fails. Three layers:

1. **consistency** — every clip came from the same spec, pose model, rtmlib
   build and person-selection policy. This catches the failure the plan calls
   the highest risk: two datasets extracted slightly differently. It never
   raises on its own and only surfaces as bad BLEU weeks later.
2. **numbers** — missing-part rates, frozen tracks, out-of-frame coordinates,
   and an automated mirror test. In COCO-WholeBody `left_shoulder` (index 5) is
   the *signer's* left, which for a frontal camera sits on the **right** of the
   image. Auslan distinguishes dominant from non-dominant hand, so a swap is a
   different sign, not a cosmetic bug. How many clips fail it decides what it
   means:
   - **A whole dataset/subset** (over `--max-mirrored-rate`, 2%): the pipeline
     flips x somewhere. The run fails; do not train.
   - **Scattered clips**: the chosen person has their back to the camera, is
     turned far enough side-on that the pose model swaps left and right, or is
     not the signer at all. Their hand labels cannot be trusted, so
     `--exclude-out` lists them and `data.exclude` leaves them out of training.
     Without `--exclude-out` they still fail the run, so nothing trains on them
     by accident. The first full extraction had 131 of 25,109 (0.5%).
     `--overlay-dir checks/mirrored --overlay-flag MIRRORED_OR_BACK_VIEW`
     renders a sample of them to look at.

   A video that is itself mirrored does not show up in this test: the person in
   it still faces the camera, and the model labels them consistently.
3. **eyes** — overlays for the worst clips plus a random sample. Orange must be
   on the signer's left hand.

Only then run the full extraction (shardable with `--shard/--num-shards`).

**Budget for it.** Measured on this machine (8-core Apple silicon, 4
performance cores), `mode=lightweight` on CPU runs at **6.6 fps**:

| corpus | clips | frames | single-process |
|---|---|---|---|
| MM-WLAuslan (train+val+test_STU) | 51,440 | 4.12 M | 173 h |
| Auslan-Daily Communication | 14,040 | 0.70 M | 30 h |
| Auslan-Daily News (estimated) | 11,069 | 1.33 M | 56 h |
| **total** | | **6.15 M** | **259 h** |

That is eleven days of laptop, and sharding will not divide it cleanly because
onnxruntime already spreads one process over roughly three threads. Two ways
out, in order of preference:

1. Extract on a CUDA box (`--device cuda`, with `onnxruntime-gpu`). This is
   embarrassingly parallel per clip and the manifest shards cleanly.
2. Stage the work. **Arm A needs only Auslan-Daily** — 30 h for Communication,
   about a day at 2–3 shards, and the baseline can be running while
   MM-WLAuslan grinds through in the background. Arms B and C are the only ones
   that need the isolated corpus.

**`--device mps` does not work** and the script now refuses it. onnxruntime maps
it to the CoreML execution provider, whose YOLOX partition cannot handle the
dynamic-shaped detection output when a frame contains no person. It does not
fail on frame one — it fails on the first frame where nobody is detected, which
in a 70k-clip run is hours in.

**Batched pose inference (`--pose-batch`, default 32).** The throughput above
is rtmlib's per-frame path: one pose-model call per person per frame. On a GPU
that is a stream of tiny calls, and adding worker processes mostly makes the
card switch between more CUDA contexts. Going from 8 to 14 workers on the A100
bought 23%. In a local decode test, running more processes than cores cut
throughput by 31%. So keep `NUM_WORKERS` at or below the vCPU count
(`os.cpu_count()` on Colab).

RTMW's ONNX export has a dynamic batch dimension, so crops from many frames
now go through one call. YOLOX's export is fixed at batch 1 and still runs per
frame. Every crop is still prepared and decoded by rtmlib's own
`preprocess`/`postprocess`, and rtmlib's fallback ("nobody detected: use the
whole image as the box") is kept. `--pose-batch 1` is the upstream path
exactly.

`verify_batching.py` checks equivalence. It ran locally on CPU against 5 real
clips (4 of them with 2–3 detected people per frame) plus synthetic blank-frame
and two-person cases. Stored keypoints matched the per-frame path to 0.00 px.
Exactly one difference turned up, in 1 of 12,103 (crop, keypoint) pairs on
news 78_191. It was a SimCC argmax tie: the top two bins differed by 1.2e-7,
and the keypoint moved by one bin. It sat on a person who was not selected,
and on a keypoint the model does not use. Any two numerical implementations
disagree this way, so the verifier judges the data training sees (model input
must agree to 1e-5) rather than bit-identity.

Speedup on CPU is negligible (1.08×). The GPU speedup, and GPU equivalence,
have to be measured on Colab:

    python verify_batching.py --from-manifest "$MANIFEST" --n 6 --device cuda

Mixing clips extracted before and after batching is fine. `pose_batch` is
recorded in each clip's metadata, but it is deliberately not one of
`verify_pose.py`'s identity fields.

**Choosing the signer (`--person-select`, default `largest`).** A Signer-Only
crop is cut around the annotated signer, but other people are often inside it
too. Auslan-Daily's own Signer Detection table lists at least one non-signer
for 88% of Communication clips and 43% of News clips. The extractor has to pick
one detection per frame, and the rule matters.

`largest` takes the biggest detector box in each frame. Because the crop is
built around the signer, the signer dominates it. This was measured against the
dataset's official pose annotation (`Pose Annotation/*.pt`), which marks the
signer in every frame. The annotation is in original-frame coordinates. The
offset into each crop is recovered per clip by consensus: a candidate that is
the signer differs from the annotation by a pure translation, so only
candidates whose keypoints agree on one shift get a vote. Results on video 46:

| | held-out: 49 clips, 3,587 frames | dev: 38 high-risk clips, 2,439 frames |
|---|---|---|
| `largest` | **99.97%**, no clip wrong | **99.96%**, no clip wrong |
| `first` (upstream: detection 0 every frame) | 98.75% | 89.13% |
| `track` (the previous default) | 97.99%, 1 clip entirely wrong | 88.97%, 2 clips entirely wrong |

The held-out clips were never looked at while the rule was being chosen.
`track`'s errors are not scattered. It commits to detection 0 in the first
frame and follows that person, so when detection 0 is a bystander the whole
clip comes out wrong. Two other, more intuitive rules were tried on the same
data and rejected:

- Excluding big boxes that contain another person scored 60.7%. The signer,
  standing in front, is often exactly that box.
- Choosing by hand activity and face visibility scored 43.4%.

What `largest` cannot fix is a detector box that swallows two people. The
biggest box is then the merged one, and the pose model fits one skeleton across
both bodies. No candidate is the signer in those frames, so no selection rule
helps. On video 46 this was 35 of about 6,000 frames. News could not be
measured, because its annotation archive was quota-blocked on Drive.

Each `.npz` now also stores `n_people` (detections per frame) and `chosen_box`
(the selected box, normalised; detector boxes can run slightly past the frame
edge). This lets a clip's selection be re-examined later without running
inference again.

**Re-extract anything extracted with `track`.** Old clips cannot be patched,
because they do not record how many people each frame had. `person_select` is
one of `verify_pose.py`'s identity fields, so a set that mixes the two policies
fails the consistency check. That is deliberate. `extract_pose.py` now
re-extracts an existing `.npz` whose spec or policy differs from the current
run, instead of skipping it.

On Colab, before running the extraction, rename the old checkpoint folder on
Drive (`auslan_work/pose` → e.g. `pose_track_old`). The notebook then starts
from an empty folder instead of restoring the old clips, and the old data is
kept rather than deleted.

### 4. Arm A, the baseline

On Colab, use `colab_train.ipynb`. It runs everything below for you:
- It downloads the Uni-Sign code, mT5 and the pre-trained weights, pinned and
  checked by sha256.
- It unpacks the poses and writes the run config with the real model and
  `data.exclude` already set.
- It trains the runs set by `LR` and `SEEDS` with `--resume`, checkpointing
  to Drive. The first baseline, 3e-5 with seeds 0 and 1, is kept as the
  reference: the gap between its two seeds is the noise floor that any later
  change has to beat.
- It scores every finished run under each decoding setting in `DECODES`
  (`train.py --eval-only`, see below) and prints one table per subset.

The config's `decode` block adds generation options on top of plain beam
search, for example `no_repeat_ngram_size` and `repetition_penalty`. It is not
part of the resume fingerprint, since it changes how a model is decoded, not
what it learns. `--eval-only` decodes the validation split with a finished
run's `checkpoint.pt` and writes `<out>/eval_<tag>/`, leaving the run's own
results alone:

```bash
python train.py --config configs/arm_a.yaml --eval-only --eval-tag nr3 \
                --set decode.no_repeat_ngram_size=3
```

`test_decode.py` checks both, including that no decode options gives exactly
Uni-Sign's own generate() call.

Locally:

```bash
python train.py --config configs/arm_a.yaml --resume \
                --set data.exclude=checks/excluded.txt
```

Auslan-Daily only. Every other arm is read as a delta against this.

### 5. Arms B and C

```bash
python train.py --config configs/arm_b_stage1.yaml --resume
python train.py --config configs/arm_b_stage2.yaml \
                --init-from runs/arm_b_stage1/checkpoint.pt --resume

for w in 0.1 0.2 0.3; do
  python train.py --config configs/arm_c.yaml \
                  --set optim.aux_weight=$w --out runs/arm_c_w$w --resume
done
```

Arm B's stage order is fixed. Running MM-WLAuslan after Auslan-Daily leaves the
model's last impression of "a clip" being a citation form — the contamination
the staging exists to prevent.

Training prints the effective learning rate per parameter group before the first
step. Check it: a stage that trains nothing still produces a loss curve, and a
flat one is easy to read as convergence.

### Checkpoints and resuming

Colab reclaims runtimes without warning; one was removed here after about eight
hours. Training therefore checkpoints itself and resumes exactly. **Always run
`train.py` with `--resume`.** On a first run it simply starts; after a kill, the
same command continues from the newest checkpoint.

- **What is saved.** The model, the optimiser, the step, the position inside
  the current epoch and inside the auxiliary stream, and every RNG. A save
  happens every `checkpoint.every_minutes` (default 30), at each epoch
  boundary, and on Stop. Each save is written to a temp file and renamed into
  place, so a kill during a save never leaves a truncated checkpoint.
- **Exact.** `python test_resume.py` hard-kills and Stop-interrupts Arms A,
  B stage 1 and C mid-epoch, resumes them, and requires the final model to be
  bit-identical to an uninterrupted run. All six cases pass. With worker
  processes the data order is still exact; only the random temporal
  subsampling of long clips draws differently after a resume.
- **Guarded.** Starting over existing checkpoints without `--resume` is
  refused; `--fresh` moves them aside instead. Resuming is also refused if the
  config, the dataset sizes, `--init-from` or `--smoke-steps` changed, because
  that would splice two runs together.
- **Size and retention.** A Uni-Sign checkpoint is about 7 GB: 588M
  parameters plus AdamW's two moment buffers. Put `--out` (or
  `checkpoint.dir`) on Drive so it outlives the runtime. `checkpoint.keep: 2`
  keeps the newest plus one older checkpoint, about 14 GB per run, and briefly
  21 GB while a new one is written before the oldest is pruned. The older one
  is a fallback in case the newest was saved after training had already gone
  wrong (a loss gone to NaN, a bad learning rate). To use it, delete the newest
  `ckpt_step*.pt` and run the same command with `--resume`: it continues from
  the remaining one.
- **Fixed along the way.** Arm C's auxiliary stream used `itertools.cycle`,
  which caches the first pass and replays it. The aux data was therefore never
  reshuffled after the first pass, and every aux batch was held in memory. It
  is now reshuffled every aux epoch.

### 6. Arm D, only if the OOV rate says so

```bash
python stitch.py --manifest data/manifest.jsonl --npz-dir data/pose \
  --out-npz-dir data/pose_stitched --out-manifest data/stitched.jsonl \
  --n-sentences 20000
```

Then add `stitched` to Arm C's aux datasets. `stitch.py` refuses to label its
output anything but `train`.

## Testing the harness before the checkpoint arrives

Every config ships with `backend: smoke`, a small real encoder-decoder whose
module names match the group patterns and which shares its left/right encoder
exactly as the real model does. It runs the whole pipeline — batching, freezing,
loss mixing, checkpoint chaining, decoding, scoring — on CPU in about a minute.
It is a harness test; its numbers are not results. Switch `backend.name` to
`unisign` when the checkpoint is in hand.

## Where the plan document is wrong

Three things in the original plan do not survive contact with the code:

* **"Four per-part encoders, weights not shared."** `models.py` does
  `gcn_modules['left'] = gcn_modules['right']` and the same for `proj_linear`.
  Four branches, three sets of weights. The hands share an encoder.
* **"RTMPose-x via MMPose."** The shipped extraction is `rtmlib.Wholebody` at
  `mode="lightweight"`, which is RTMW-l-m at 192×288 plus YOLOX-tiny. No mmpose.
* **"Normalisation is root subtraction; the body is not normalised."** The body
  *is* normalised — `crop_scale` maps it into [-1,1] using one bounding box over
  the whole clip — and the hands and face are root-subtracted and then divided
  by that same body scale. The parts end up on one common scale.

Also corrected here: the face group is jaw(9), inner lip(8), **nose tip last**.
The nose is the face graph's centre node and its root. Ordering is load-bearing
because each part is an ST-GCN over a fixed graph.

`spec.py`'s port of `load_part_kp`/`crop_scale` is checked numerically against
the repo's own implementation — max absolute difference 3e-08 over random
clips — and a real batch has been pushed through the repo's actual ST-GCN
modules to confirm the node counts, ordering and 1024-dim concatenation.

## Things this code will not do for you

**Grammatical non-manual markers.** The face group is 9 jaw points, 8 inner-lip
points and the nose tip. Mouthing and head pose are in; eyebrows and eyes are
not. Brow-marked polar questions, wh-questions, topicalisation and conditionals,
and gaze-based reference have no input channel. Fine-tuning does not fix it —
see `FACE_UNMODELLED` in `spec.py`. Say so in the requirements before anyone is
promised question/statement disambiguation.

**A respectable BLEU.** YouTube-SL-25 gets 15.4 BLEU on How2Sign off 1394 hours
of ASL pre-training. This project has 45 hours of continuous Auslan and a base
model that saw CSL and ASL but no Auslan. Single digits are the expected
outcome. Agree that at kickoff, not at delivery.

**Averaging the two Auslan-Daily subsets.** `evaluate.py` has no code path that
does it. Communication is multi-signer in-the-wild dialogue; News is one studio
presenter. A mean over them describes neither.

## If you change spec.py

Change it once, in that one file, and bump `SPEC_VERSION`. Every stored clip
carries the fingerprint, so a mismatch is caught at load rather than absorbed.
