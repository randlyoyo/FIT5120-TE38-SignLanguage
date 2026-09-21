# auslan_sft: English → Auslan pose → stick-figure video

This project turns English text into an Auslan pose sequence. It is built on a **pretrained sign-language model**, transfer-learned to Auslan with conservative supervised fine-tuning (SFT), and the output is rendered as a stick-figure MP4.

**Read [PRETRAINED_MODEL_ANALYSIS.md](PRETRAINED_MODEL_ANALYSIS.md) first.** In short:

- The Progressive Transformers checkpoint is no longer downloadable: its Dropbox link was removed and is disabled.
- No other public English text→pose SLP checkpoint exists.
- So the generator is **initialised from the Uni-Sign OpenASL pose-only checkpoint**, which is ASL, English, and uses the same rtmlib keypoints as our Auslan data:
  - mT5 encoder and decoder blocks: **390.3 M parameters loaded, 99.94 %**;
  - only **0.245 M** new parameters: pose in/out heads and a progress counter;
  - Uni-Sign's frozen ASL-trained pose encoder provides a feature loss.
- Progressive Transformers remains the *architectural template*: autoregressive continuous frames plus a progress counter.
- License: **CC BY-NC 4.0**, inherited from Uni-Sign. Non-commercial use only.

```
English ─► mT5 SentencePiece ─► mT5 encoder (pretrained) ─► mT5 decoder blocks (pretrained, causal)
                                                                 ▲ pose_in (new)      │ pose_out + counter (new)
                                                          previous frame + counter ◄──┘
        ─► pose (T, 79, 2) ─► stick-figure renderer ─► MP4
```

Pretrained sign-language knowledge (ASL/CSL motion features, English semantics, sign-aligned sequence modelling) is **not** Auslan knowledge. Every Auslan-specific mapping comes from your fine-tuning data. A model that reaches low coordinate error has not thereby been shown to sign correct Auslan (see §Evaluation).

## Layout

```
auslan_sft/
├── PRETRAINED_MODEL_ANALYSIS.md
├── configs/auslan_sft.yaml
├── preprocessing/
│   ├── extract_pose.py                video → raw 133-keypoint .npz (rtmlib, same as Uni-Sign)
│   ├── normalize_pose.py              raw .npz → normalised, gap-filled, 25 fps .pt
│   ├── make_splits.py                 seeded video-level train/val/test (.csv + .txt)
│   └── annotations_from_manifest.py   Auslan-Daily adapter (reuses ../unisign extractions)
├── slp_models/
│   ├── skeleton.py        79-joint layout, bones (single source of truth)
│   ├── pretrained_slp.py  text encoder, Uni-Sign pose encoder, generator, checkpoint loader + gate
│   ├── pose_decoder.py    Progressive-Transformer decoder on mT5 blocks
│   ├── freeze.py          modes A/B/C, differential learning rates
│   └── losses.py          masked pose / velocity / acceleration / bone / counter / feature losses
├── slp_data/auslan_dataset.py   dataset, normalisation stats, augmentation, collate
├── slp_utils/                   config, checkpoint, metrics, visualization,
│                                unisign_format (differentiable converter), back_translation
├── verify_checkpoint.py   train.py   evaluate.py   infer.py   render_pose.py
├── tests/test_pipeline.py
└── requirements.txt
```

The packages are named `slp_models` / `slp_data` / `slp_utils`, not `models` / `datasets` / `utils`. Uni-Sign ships top-level `models.py` and `utils.py`, and HuggingFace has a `datasets` package; plain names would shadow them.

## Pose representation

**79 joints × (x, y)**, from the rtmlib `Wholebody` COCO-WholeBody estimator. This is the estimator the pretrained model was trained on, and the reason MediaPipe is not used.

| Positions | Part | COCO-WholeBody indices |
|---|---|---|
| 0–8 | body: nose, ears, shoulders, elbows, wrists | 0, 3–10 |
| 9–29 | signer's left hand | 91–111 |
| 30–50 | signer's right hand | 112–132 |
| 51–68 | face: jaw 9 + inner lip 8 + nose tip | 23…39 (step 2), 83–90, 53 |
| 69–78 | eyebrows (extra; Uni-Sign does not model them) | 40–49 |

Nothing is ever mirrored, because Auslan handedness is linguistic. There are no hip keypoints, so the renderer draws a fixed torso outline hanging from the shoulders.

## Normalisation (exact)

Per clip. The constants are computed once for the whole clip, never per frame:

```
p_px[t,j]  = (x[t,j]·W, y[t,j]·H)                                  undo per-axis frame scaling
valid[t,j] = score[t,j] > 0.3  and  inside the frame ± 25 %
c   = median_t ( (p_px[t,L_sh] + p_px[t,R_sh]) / 2 )               over frames with both shoulders valid
s   = median_t ||p_px[t,L_sh] − p_px[t,R_sh]||
p̂   = (p_px − c) / s                                               units: shoulder widths, y down
```

- **Missing keypoints.**
  - Internal gaps up to `max_gap_sec` (0.2 s) are linearly interpolated and count as valid.
  - Longer gaps are interpolated for input continuity but masked out of every loss and metric.
  - Leading and trailing gaps hold the nearest value and are masked.
  - A joint never seen in the clip is NaN, replaced by the training mean pose, and masked.
- **Resampling** is linear to `target_fps` (25). A resampled joint is valid only if both neighbouring source frames were.
- **Rejected clips** are those with shoulders visible in fewer than 50 % of frames, no hands in 90 % of frames, or fewer than 8 or more than 400 frames. They are listed in `pose_dir/rejected.csv`.
- **Stored per clip** (`norm`): `center_px`, `scale_px`, `W`, `H`, `source_fps`. Uni-Sign's own `crop_scale` box is stored too, which makes the conversion back to Uni-Sign input exact. The test shows a 2.4e-07 difference against `unisign/spec.py::load_part_kp`.
- **Model standardisation.** A per-joint mean and std (std floor 0.02) are computed on the **train split only**. They are saved to `<output_dir>/norm_stats.json` and inside every checkpoint, together with the dataset view bounds and a canonical camera used for rendering and back-translation.

## Training modes

| Mode | Trainable (default config) | Params trainable |
|---|---|---:|
| A | pose_in / pose_out / counter_out (+ optionally the last k decoder blocks) | 0.245 M |
| **B (default)** | the above + decoder blocks 8–11 + final norm + encoder blocks 10–11 + final norm | 52.2 M |
| C | everything except the 192 M embedding matrix; pretrained learning rate × 0.1 | 198 M |

- Learning rates: pretrained 1e-5, new 1e-4.
- AdamW, weight decay 0.01 (none on norms or biases), T5 dropout 0.1, gradient clip 1.0.
- Warmup 5 %, then cosine decay to 5 %.
- Early stopping on validation **free-running** DTW, patience 8. `best.pt` and `latest.pt` both carry the model, optimizer, scheduler, epoch, metric, config and pose representation with its normalisation stats.
- AMP: bf16 by default (T5 overflows in fp16). DDP through `torchrun`.

**Loss:**

```
λ_pose·(SmoothL1 + wrist-relative hand) + λ_vel·L_vel + λ_acc·L_acc + λ_bone·L_bone + λ_counter·L_counter + λ_feat·L_feat
```

- Padded frames and invalid joints contribute exactly zero (tested).
- `L_feat` is the relative MSE between frozen Uni-Sign pose-encoder features of the generated and ground-truth poses. That encoder is verified identical to `Uni_Sign`'s pose branch, max difference 0.
- `pose_loss_type: mse` reproduces Progressive Transformers' loss.

**Augmentation.** Coordinate noise 0.005, global scale ±5 %, duration ±10 %, 3 % frame dropout. Horizontal flip is **off** because it swaps handedness. Decoder inputs receive Progressive Transformers' Gaussian noise.

## Evaluation and its limits

`evaluate.py` generates free-running output for a split and reports:
- DTW distance;
- DTW-aligned MPJPE, overall and for body / hands / face;
- wrist-relative hand-shape error;
- resampled MPJPE and velocity error;
- PCK@0.1 and PCK@0.2 (in shoulder widths);
- motion-energy ratio and length ratio.

Every metric is also computed for a **static mean-pose baseline**. Side-by-side videos are rendered for the first N samples.

- **What these metrics cannot tell you.** Coordinate metrics reward staying near one reference signer's trajectory. They cannot tell which differences change meaning (handshape, location, movement, orientation, non-manual features) from those that do not (body size, speed, style). Autoregressive regression models tend to collapse toward an average pose, which can still score a "reasonable" MPJPE. A motion-energy ratio far below 1, or failing to beat the mean-pose baseline, exposes that collapse.
- **Semantics.** `--slr-checkpoint` back-translates both the generated and the ground-truth poses with a Uni-Sign SLT model, such as the Auslan-Daily runs in `../unisign`, and reports BLEU/chrF for both. The interface is `slp_utils/back_translation.py::SLRBackTranslator`. Only a clip the SLT model never trained on is a valid test. Final claims about Auslan quality need review by Deaf Auslan signers.

## Commands, in order

The paths below are the defaults in `configs/auslan_sft.yaml`. Every key can be overridden with `--set key=value`.

### Step 1: environment installation

```bash
cd auslan_sft
python -m venv .venv && source .venv/bin/activate          # or use the repo's venv / Colab
pip install -r requirements.txt
# Colab/CUDA: pip install --no-deps rtmlib==0.0.16 && pip install onnxruntime-gpu
# Pretrained assets (already present in this repo):
#   ../checkpoints/openasl_pose_only_slt.pth   https://huggingface.co/ZechengLi19/Uni-Sign (rev eab251b7)
#   ../Uni-Sign                                 https://github.com/ZechengLi19/Uni-Sign (commit eed438b)
#   ../Uni-Sign/pretrained_weight/mt5-base      https://huggingface.co/google/mt5-base
python tests/test_pipeline.py --video /path/to/any_signing_clip.mp4      # optional local self-test (CPU)
```

### Step 2: dataset preprocessing

Generic dataset (`data/videos/*.mp4` plus `data/annotations.csv` with columns `video_path,text`):

```bash
python preprocessing/extract_pose.py   --config configs/auslan_sft.yaml --device cuda    # add --shard i/n to parallelise
python preprocessing/normalize_pose.py --config configs/auslan_sft.yaml
```

Auslan-Daily: the poses were already extracted by `../unisign`, so skip extraction.

```bash
# restore the unisign .npz files (Drive: auslan_work/pose/chunk_*.tar) into data/raw_pose/, then
python preprocessing/annotations_from_manifest.py --manifest ../data/manifest.jsonl \
    --out data/annotations.csv --exclude /path/to/auslan_work/excluded.txt
python preprocessing/normalize_pose.py --config configs/auslan_sft.yaml --set data.id_column=uid
```

### Step 3: train/val/test split

```bash
python preprocessing/make_splits.py --config configs/auslan_sft.yaml
# Auslan-Daily: keep the official split (needed if you back-translate with the ../unisign SLT models):
python preprocessing/make_splits.py --config configs/auslan_sft.yaml --set data.id_column=uid split.method=column
# many repeated sentences? group them so no sentence crosses splits:
#   --set split.group_column=text
```

### Step 4: verify pretrained checkpoint loading (do not skip)

```bash
python verify_checkpoint.py --config configs/auslan_sft.yaml --forward
```

Expected output, measured on this machine:

```
Loaded pretrained parameters: 390,315,264 (281 tensors, 99.94% of the generator)
Missing parameters: 0
Unexpected parameters: 0
Deliberately ignored checkpoint tensors: 3 [... encoder.embed_tokens, decoder.embed_tokens, lm_head]
Newly initialised (by design): 6 tensors [pose_in, pose_out, counter_out weights/biases]
Frozen Uni-Sign pose encoder tensors loaded: 343
Trainable parameters: 52,163,487 / Total parameters: 390,560,415
VERIFIED: pretrained weights loaded; safe to start SFT.
```

`train.py` prints the same block and **stops**, before any training step, if any of the following hold:
- a mapped tensor is missing or has the wrong shape;
- the checkpoint contains unexpected tensors;
- the loaded fraction is below `pretrained.min_loaded_fraction`.

### Step 5: SFT training

```bash
python train.py --config configs/auslan_sft.yaml                                   # mode B (default)
python train.py --config configs/auslan_sft.yaml --set training.mode=A training.output_dir=checkpoints_A
python train.py --config configs/auslan_sft.yaml --set training.mode=C training.output_dir=checkpoints_C
python train.py --config configs/auslan_sft.yaml --resume                          # after a disconnect
# Auslan-Daily ids: add  --set data.id_column=uid split.method=column
# multi-GPU: torchrun --nproc_per_node 2 train.py --config configs/auslan_sft.yaml
```

- Logs go to stdout and `checkpoints/metrics.jsonl`. Set `training.tensorboard: true` for TensorBoard.
- Checkpoints are about 1.8 GB each: the frozen 192 M embedding matrix is saved as well.

### Step 6: evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt --split test
# with back-translation (Uni-Sign SLT checkpoint trained on a split that excludes these clips):
python evaluate.py --checkpoint checkpoints/best.pt --split test --slr-checkpoint /path/to/unisign_run/checkpoint.pt
```

### Step 7: English-to-Auslan inference

```bash
python infer.py --checkpoint checkpoints/best.pt --text "I am going to university today." --output outputs/demo.mp4
# writes outputs/demo.npy (T,79,2), outputs/demo.json, outputs/demo.mp4
```

### Step 8: render stick-figure video

```bash
python render_pose.py --pose outputs/demo.npy --output outputs/demo_rerender.mp4
python render_pose.py --pose data/pose/<sample_id>.pt --output outputs/gt.mp4          # ground truth
python render_pose.py --pose outputs/demo.npy --compare data/pose/<sample_id>.pt --output outputs/side_by_side.mp4
```

## What has been verified, and what has not

Verified locally (CPU, torch 2.14, transformers 4.57.6) by `tests/test_pipeline.py` and the checks above:
- extractor output is bit-identical to `../unisign/extract_pose.py`;
- the Uni-Sign converter matches `spec.load_part_kp` (2.4e-07);
- normalisation round-trips;
- the loss masks work;
- the real checkpoint loads with 0 missing and 0 unexpected tensors;
- the frozen feature encoder is identical to `Uni_Sign`'s pose branch;
- the decoder is causal;
- a small model can overfit and generate the target motions and lengths;
- the full pipeline runs end to end: split → normalise → verify → train (mode B) → resume → evaluate → infer → render;
- in mode B, frozen tensors stay bit-identical;
- one full-size (390 M) mode-B training epoch runs;
- 2-rank DDP (gloo) trains.

**Not yet verified:**
- training on real Auslan data, and whether it converges or beats the mean-pose baseline;
- the CUDA / bf16 path and NCCL;
- transformers 5.x (the constructor was checked against the source only);
- end-to-end back-translation. The converter and loader are tested separately, but the combination has not been run.
