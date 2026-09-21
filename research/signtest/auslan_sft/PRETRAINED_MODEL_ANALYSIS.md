# Pretrained model analysis for English → Auslan pose SFT

Checked on 2026-09-17. Every claim below is marked **verified**, meaning I ran it, downloaded it or read the code, or **unverified**, meaning it comes from a paper or README and I did not reproduce it.

---

## 0. Summary

| Candidate (public SLP / text-to-pose) | Usable checkpoint? | Evidence |
|---|---|---|
| **Progressive Transformers** (Saunders et al., ECCV 2020), `BenSaunders27/ProgressiveTransformersSLP` | **No** | The README once linked `PreTrained_PTSLP_Model.ckpt` on Dropbox (added in commit `e325ff5`, 2021-05-21). The link was removed from the README in the last commit, `ff8b28c` (2024-01-22). Downloading it today returns a 92 KB Dropbox HTML page, *"Error"* / `error_pages__disabled.js`. Issue #36, *"Pre-trained model download link is disabled"* (Dec 2021), has no maintainer reply, and neither does #56, *"Could you please share your pretrained model?"*. The repo has no releases. **verified** |
| **SOKE / Signs as Tokens** (ICCV 2025), `2000ZRL/SOKE` | **Generator: no.** Tokenizer only. | The README links only the *decoupled tokenizer* (SMPL-X VQ-VAE) checkpoint, plus mean/std. `last.ckpt` of the autoregressive generator is not released: issue #2 is open with no answer. The representation is SMPL-X parameters, so every Auslan video would first need SMPL-X fitting. License: CC BY-NC-ND 4.0 (**no derivatives**), which forbids exactly the fine-tuned derivative we want. **verified** (README, license.txt, issue) |
| **Sign-IDD** (AAAI 2025), `NaVi-start/Sign-IDD` | No | The README has no checkpoint links. It takes gloss input and uses PHOENIX14T in the Progressive Transformers format. **verified** (README) |
| `sign-language-processing/transcription` text_to_pose | No usable English→pose model | Input is HamNoSys or SignWriting, not English. The README documents no released weights. **verified** (README, code tree) |
| General whole-body text-to-motion (HumanTOMATO, MotionCraft) | Not a sign-language model | These are SMPL-X, trained on Motion-X gestures rather than signing. They would need SMPL-X fitting, and they carry no signing priors. Not pursued. |

**Conclusion.** No public, downloadable text-to-pose SLP checkpoint exists that can be fine-tuned into an English → Auslan keypoint generator. The Progressive Transformers checkpoint *did* exist and is gone. Even if it could be recovered, it would transfer little. Per `Configs/Base.yaml` it is a 2-layer, 512-d model of roughly 13 M parameters, estimated from that config rather than loaded:
- about 4 M in an encoder over a 1,089-type **German DGS gloss** vocabulary, which English text cannot use;
- about 8 M in a decoder tied to a 50-joint 3D IK skeleton with no face (§6).

Per the brief, the fallback is **not** a silently random Transformer. The fallback is:

> **Selected pretrained model: Uni-Sign pose-only SLT checkpoint fine-tuned on OpenASL** (`openasl_pose_only_slt.pth`, HuggingFace `ZechengLi19/Uni-Sign`).
>
> It is the closest *publicly downloadable* pretrained **sign-language** model that meets both conditions below.
> 1. It is English-language.
> 2. It operates on the **same whole-body keypoint layout the Auslan data is already extracted in**, including 21 points per hand (rtmlib RTMW COCO-WholeBody).
>
> It is a **sign → text** model, not a production model. We run it "in reverse". This is an honest architectural transfer, not a claim that Uni-Sign already produces poses.

The **Progressive Transformer is kept as the architectural template**: a continuous autoregressive pose decoder with a progress counter, teacher forcing and Gaussian input noise. It is instantiated **with transformer blocks whose weights come from the Uni-Sign checkpoint**.

---

## 1. Selected pretrained model

`openasl_pose_only_slt.pth`: Uni-Sign (Li et al., *Uni-Sign: Toward Unified Sign Language Understanding at Scale*, ICLR 2025), in its pose-only configuration.

- **Code:** `Uni-Sign/` (commit `eed438b`), `models.py::Uni_Sign`.
- **File:** 1,186,924,838 bytes, sha256 `f836ea66bc837bbe6ed717a4b9bece87875f03ef96d4bf1092ca3dd767982798`, HF revision `eab251b7`. **verified**
- **Format:** `{'model': state_dict}`, 627 tensors, stored in **bfloat16** (the int64 values are ST-GCN buffers). **verified**

Parameter groups, measured with `torch.load(mmap=True)` (**verified**). Tensors are counted as stored, so the tied duplicates and the shared left/right hand branch are counted twice:

| Prefix | Tensors | Params | Role in Uni-Sign |
|---|---:|---:|---|
| `proj_linear.*` | 8 | 0.00 M | (x, y, conf) → 64 per part |
| `gcn_modules.*` | 152 | 0.55 M | spatial ST-GCN per part (the left hand shares the right hand's weights) |
| `fusion_gcn_modules.*` | 180 | 5.55 M | temporal ST-GCN per part |
| `part_para`, `pose_proj.*` | 3 | 0.79 M | part fusion → 768 |
| `mt5_model.shared` | 1 | 192.09 M | mT5 SentencePiece embedding (250,112 × 768) |
| `mt5_model.encoder.*` | 111 | 277.04 M (≈85 M blocks + the tied duplicate `embed_tokens`) | mT5-base encoder, 12 layers |
| `mt5_model.decoder.*` | 171 | 305.36 M (≈113 M blocks + the tied duplicate `embed_tokens`) | mT5-base decoder, 12 layers, with cross-attention |
| `mt5_model.lm_head` | 1 | 192.09 M | untied vocabulary head |

How far the checkpoint moved from `google/mt5-base` (relative Frobenius change, **verified**). This shows the mT5 weights really were trained on sign data, not merely copied:

| Tensor | ‖Δ‖/‖W₀‖ |
|---|---:|
| `shared.weight` | 0.003 |
| `encoder.block.0 …SelfAttention.q` | **1.353** |
| `encoder.block.11 …wo` | 0.233 |
| `decoder.block.0 …EncDecAttention.k` | 0.295 |
| `decoder.block.11 …wo` | 0.274 |
| `lm_head.weight` | 0.953 |

The encoder's first layer changed by more than its own norm, because it learned to read pose features. See the risk note in section 12.

## 2. Original training data

- **OpenASL** (Shi et al., 2022): about 288 h of ASL video from online news and VLOG sources, with English translations and more than 200 signers. **unverified** (paper figures)
- Uni-Sign's pipeline pre-trains on **CSL-News** (Chinese Sign Language, about 1,985 h) before downstream fine-tuning. The HF model card excerpt I could fetch does not state the initialisation of this particular file, so the CSL-News pre-training → OpenASL fine-tuning chain is **unverified for this file**.
- mT5-base is pre-trained on mC4 text in 101 languages. **verified** (config, `google/mt5-base`)

## 3. Original language

- Signed: **ASL** (American Sign Language), plus CSL through pre-training if the chain above holds.
- Spoken/written: **English** output.
- **No Auslan.** ASL and Auslan are unrelated sign languages. Auslan belongs to BANZSL, with BSL and NZSL. They differ in their manual alphabets (Auslan fingerspells two-handed, ASL one-handed), in most of the lexicon, and in their mouthing conventions.

## 4. Model input (original)

- `src_input[part]` for part in `body (9)`, `left (21)`, `right (21)`, `face_all (18)`. Each is a `[B, T, J, 3]` tensor of (x, y, confidence).
- Normalisation is Uni-Sign `crop_scale`: one bounding box per clip over the body joints maps the body to [-1, 1]. Hands and face are made relative to the wrist or nose tip and divided by the same scale. Joints with confidence ≤ 0.3 are zeroed.
- An mT5 text prefix `"Translate sign language video to English: "` is prepended in embedding space. **verified** (`models.py`, `unisign/spec.py` port, max error 3e-08)

## 5. Model output (original)

- English token sequence from the mT5 decoder with `lm_head` (label-smoothed cross-entropy 0.2, beam search).
- **No pose output head exists.** **verified**

## 6. Pose representation

| | Uni-Sign (source) | This project (target) |
|---|---|---|
| Estimator | rtmlib `Wholebody(mode="lightweight")` = RTMW-l-m 256×192 + YOLOX-tiny | **same**, already used for all 25,109 Auslan-Daily clips (`unisign/extract_pose.py`) |
| Keypoints | COCO-WholeBody 133, using 69 | the same 69 **+ 10 eyebrow points** = 79 (`slp_models/skeleton.py`) |
| Dimensions | 2D + confidence | 2D (x, y) + validity mask |
| Hands | 21 + 21 | 21 + 21 |
| Normalisation | clip bounding box → [-1, 1] | clip-level shoulder centre and shoulder width (see README §Normalisation) plus a **differentiable converter back to Uni-Sign's exact format** (`slp_utils/unisign_format.py`) |

For reference, Progressive Transformers used **gloss** input (German DGS gloss, `src_vocab.txt`: 1,089 PHOENIX14T gloss types) or German text. Its output was 150 values per frame: 50 joints × 3 from OpenPose, lifted to 3D by inverse kinematics and divided by 3, plus 1 progress counter. **verified** (`Configs/Base.yaml`, `Data/tmp`, README)

That is German weather-forecast signing with no face. Its 50 joints and 3D IK skeleton do not match any Auslan data we hold.

## 7. Checkpoint availability

- Progressive Transformers: **unavailable** (section 0).
- SOKE generator: **unavailable**, and its NoDerivatives license would forbid SFT in any case.
- Uni-Sign OpenASL pose-only: **available and already on disk**:
  - `checkpoints/openasl_pose_only_slt.pth`, with sha256 matching as above;
  - `mt5-base` tokenizer and config in `Uni-Sign/pretrained_weight/mt5-base`.
- **License: CC BY-NC 4.0 (non-commercial).** Every model fine-tuned from it inherits the non-commercial restriction.

## 8. Download location

```
https://huggingface.co/ZechengLi19/Uni-Sign/resolve/eab251b7/openasl_pose_only_slt.pth   (sha256 f836ea66…2798)
https://huggingface.co/google/mt5-base      (tokenizer + config; Uni-Sign/pretrained_weight/mt5-base is a copy)
https://github.com/ZechengLi19/Uni-Sign     (commit eed438b; stgcn_layers/ is imported, not copied)
```

## 9. Which weights are reused

The generator, `SignT5TextToPose`, reuses the following (**verified by `verify_checkpoint.py`**; the counts are printed before training):

| Our module | Loaded from checkpoint key | Params | Role |
|---|---|---:|---|
| `text_encoder.embed` | `mt5_model.shared.weight` | 192.1 M | English SentencePiece embeddings (same tokenizer as pretraining) |
| `text_encoder.stack.block.0-11`, `final_layer_norm` | `mt5_model.encoder.*` (minus `embed_tokens`) | 84.95 M | English text encoder |
| `pose_decoder.stack.block.0-11`, `final_layer_norm` | `mt5_model.decoder.*` (minus `embed_tokens`) | 113.27 M | autoregressive decoder: causal self-attention with T5 relative position bias, cross-attention to the encoder, gated-GELU FFN |
| `feature_encoder.*` (frozen, loss only) | `proj_linear`, `gcn_modules`, `fusion_gcn_modules`, `part_para`, `pose_proj` (343 tensors) | 5.35 M unique | ASL-trained sign-pose encoder, used as a **perceptual "sign feature" loss** and for sanity checks |

A test confirms the feature encoder produces exactly the same output as `Uni_Sign`'s own pose branch (max difference 0.0). The same checkpoint family can also be the **back-translation SLR evaluator**. That covers the existing Auslan-fine-tuned Uni-Sign run in `unisign/` (OpenASL init, Communication BLEU-4 18.03), via `slp_utils/back_translation.py`.

**Deliberately not loaded:**
- `mt5_model.lm_head.weight`: a vocabulary head is meaningless for pose regression.
- `mt5_model.{encoder,decoder}.embed_tokens.weight`: tied duplicates of `shared`.

## 10. Layers that must be replaced (new, randomly initialised)

| Module | Shape | Params | Why |
|---|---|---:|---|
| `pose_decoder.pose_in` | Linear(79×2 + 1 → 768) | 0.12 M | replaces the token embedding: previous frame + progress counter → d_model. Initialised so its output RMS matches the pretrained embedding RMS |
| `pose_decoder.pose_out` | Linear(768 → 79×2) | 0.12 M | replaces `lm_head`: d_model → standardised joint coordinates |
| `pose_decoder.counter_out` | Linear(768 → 1) | 769 | Progressive Transformers progress counter / stop signal |

The new parameters are 245,151 out of 390,560,415 in the generator, **0.063 %**. Measured load: **390,315,264 parameters loaded (99.94 %), 0 missing, 0 unexpected**; mode B trains 52,163,487. The loader refuses to train if the pretrained fraction falls below `pretrained.min_loaded_fraction` (default 0.95), or if any tensor it intends to load is missing or has the wrong shape.

## 11. Domain gap: ASL/CSL (source) → Auslan (target)

**What is plausibly transferable.** This is *pretrained sign-language knowledge*:
- mT5's English subword semantics, and an encoder trained to organise meaning for a sign-conditioned decoder.
- Decoder self-attention and FFNs trained to produce **sign-aligned, left-to-right sequences** conditioned by cross-attention. This is a sequence-modelling prior, **not a motion prior in coordinate space**: the Uni-Sign decoder never produced poses.
- The frozen ST-GCN pose encoder encodes **what signing looks like**: hand configurations, hand–body relations and short-range dynamics (temporal kernel 5), learned from hundreds of hours of ASL. This is the only *pose-space* sign prior available, and it enters through the feature loss.

**What is not transferable, and must come from the Auslan SFT data.** This is *Auslan-specific linguistic knowledge*:
- The lexicon (Auslan signs for English words), Auslan grammar and word order, two-handed fingerspelling, spatial reference, non-manual markers and English mouthing patterns.
- Nothing from ASL tells the model which Auslan sign corresponds to an English word. Transferring an ASL model **does not produce Auslan**, and it can bias the output toward ASL-like handshapes where the Auslan data is thin.

**Other gaps:**
- Direction is reversed: the encoder was trained on pose inputs, the decoder on text outputs.
- Domain: OpenASL news and vlogs versus Auslan-Daily everyday communication and news.
- Recording conditions differ. The keypoint extractor is identical, so there is no representation gap there.
- **English text ≠ gloss.** The input is natural English, not Auslan gloss, and the model has to learn the reordering and lexical choice implicitly from sentence-level pairs. Nothing here treats English as gloss.

## 12. Exact SFT strategy

1. **Preprocess** with the same rtmlib extractor → signer-space normalisation → resample to 25 fps.
2. **Load:**
   - mT5 encoder and embeddings, plus the mT5 decoder blocks, from `openasl_pose_only_slt.pth`;
   - the new `pose_in`, `pose_out` and `counter_out`;
   - the frozen Uni-Sign pose encoder for the feature loss.
   Print loaded / missing / unexpected / trainable counts, and **stop** if the loaded fraction is below 0.95.
3. **Train** a Progressive-Transformer-style model: teacher forcing, Gaussian noise on decoder inputs, and a counter target of (t+1)/T.
4. **Mode B is the default** (partial fine-tuning):
   - Frozen: the embeddings, encoder blocks 0–9, and decoder blocks 0–7.
   - Trained: encoder blocks 10–11 plus the encoder's final norm, decoder blocks 8–11 plus the decoder's final norm, and all new heads.
   - Learning rates: pretrained 1e-5, new 1e-4. AdamW, weight decay 0.01, dropout 0.1, gradient clip 1.0, warmup 5 %, then cosine.
   - Early stopping on validation DTW from free-running generation, patience 8.
   - Mode A trains heads only (optionally plus the last decoder layers). Mode C unfreezes everything except the embedding matrix, with learning rate ÷ 10.
5. **Loss:**
   - λ_pose: SmoothL1 plus a wrist-relative hand term.
   - λ_vel, λ_acc, λ_bone.
   - λ_counter.
   - λ_feat: MSE between frozen Uni-Sign features of the generated and ground-truth poses.
   - Every term is masked by frame padding and joint validity.
6. **Ablations to run before believing the transfer helps.** Each is a config switch; none of them happens silently:
   - `pretrained.decoder_source: none`, which needs `allow_random_init: true`, measures what the mT5 decoder blocks actually contribute.
   - `pretrained.encoder_source: mt5_base` uses the original `google/mt5-base` encoder. It has not been adapted to pose inputs (block-0 attention changed by 135 % in OpenASL), and its decoder cross-attention mismatch is a known risk.
   - `loss.lambda_feat: 0`.
7. **Evaluate:**
   - DTW-MPJPE, velocity error, DTW distance, PCK and motion-energy ratio, against a static mean-pose baseline.
   - Optionally, back-translation BLEU through an Auslan SLR model.
   - Low coordinate error **does not** establish that the output is linguistically correct Auslan (README §Evaluation limits).

**Known risk.** Regressive autoregressive pose models trained with MSE/L1 tend to collapse toward the mean pose ("regression to the mean"; reported for Progressive Transformers and in later SLP literature). `evaluate.py` reports the motion-energy ratio and the mean-pose baseline so that collapse is visible instead of hidden behind a low MPJPE.
