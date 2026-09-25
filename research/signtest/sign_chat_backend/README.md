# Auslan sign-chat backend

The model side of the chat app: a user types or signs, and an avatar replies in Auslan with the English text underneath. This folder contains the inference code and an HTTP API. The frontend (chat box and avatar) calls this API.

```
user types text ─────────────────────────────────────┐
                                                     ├─> dialogue model ─> reply text ─> text→sign model ─> avatar pose (+ mp4)
user signs (video) ─> pose (rtmlib) ─> sign→text ────┘
```

| Part | Model | Trained in | Measured |
|---|---|---|---|
| sign → text | Uni-Sign pose-only: Arm A from `openasl_pose_only_slt.pth`, fine-tuned on Auslan-Daily | `unisign/` (see `HANDOFF.md`) | val BLEU-4: 18.03 on Communication, 5.20 on News (single seed) |
| text → sign | SignSparK fine-tuned on Auslan-Daily Communication, round 2 | `auslan_smplx/` (see `EXPERIMENTS.md` E26–E34) | test text-only DTW, released → fine-tuned: hand 32° → 26.5°, body 29.6° → 27.5° |
| dialogue | pluggable: local Qwen2.5-1.5B-Instruct (default), any OpenAI-compatible server, Claude API, or `echo` | – | – |

**Known limits.** None of these numbers says whether a sign is *correct*; that needs an Auslan signer. The text→sign model does best on short everyday sentences, so the dialogue model is asked for replies of 15 words or fewer. Names and places outside Auslan-Daily have no sign it can borrow. Sign→text works on everyday signing filmed from the front, with both shoulders in view.

## Quick start

**Frontend development, no GPU (mock models).** The API and file formats are the same as the real server, but the outputs are fake:

```bash
cd sign_chat_backend
pip install -r requirements.txt        # only the first block is needed for mock mode
SIGNCHAT_MOCK=1 SIGNCHAT_DIALOGUE=echo uvicorn signchat.server:app --port 8000
# open http://localhost:8000/docs
```

**Real models.** Open `colab_server.ipynb` in Colab on a GPU runtime and run all cells. It takes the weights from the project Drive (`auslan_work/...`), starts the server and prints a public `https://….trycloudflare.com` URL for the frontend. On your own GPU machine, copy `config.example.yaml` to `config.yaml`, fill in the paths, and run:

```bash
SIGNCHAT_CONFIG=config.yaml uvicorn signchat.server:app --host 0.0.0.0 --port 8000
python scripts/demo_cli.py --config config.yaml       # the same pipeline in the terminal, without HTTP
```

**Tests** (mock models, run on a laptop): `python -m pytest -q tests`

## API

All responses are JSON. Media URLs are relative (`/media/...`) unless `public_base_url` is set.

### `POST /api/chat/text`
```json
{"text": "Hello, how are you?", "session_id": "optional; omit on the first turn"}
```

### `POST /api/chat/sign`
`multipart/form-data`:
- `video`: mp4, webm, mov, mkv, avi or m4v, at most 50 MB. Only the first 20 s is used.
- `session_id`: optional.
- `mirrored`: set `true` if the video is a mirror image, which is what most front-camera recordings are. The model was trained on unmirrored video, so this matters.

Both chat endpoints return:
```json
{
  "session_id": "3f0c…",                       // send it back on the next turn
  "input": {
    "mode": "sign",                            // or "text"
    "text": "Hello, how are you?",             // what the user said / what was recognised
    "recognition": {"raw_text": "hello , how are you ?", "frames": 84, "signer_visible": 1.0, "timings": {…}}   // sign only
  },
  "reply": {
    "text": "I am good, thank you. And you?",  // show this under the avatar
    "sign": {
      "pose_url": "/media/20260925-…json",     // drive the avatar with this
      "video_url": "/media/20260925-…mp4",     // skeleton rendering; fallback or debug view
      "frames": 76, "fps": 25,
      "retrieved": "i am good thank you .",    // training sentence the handshapes came from
      "seen": false                             // true = that exact sentence is in the training data
    }
  },
  "timings": {"input": 0.0, "dialogue": 0.4, "sign": 3.1, "total": 3.5}
}
```

### Other endpoints
- `POST /api/translate/text-to-sign` with `{"text": …}`: signing only, no dialogue. Returns the `reply.sign` object above.
- `POST /api/translate/sign-to-text` with multipart `video`, `mirrored`: recognition only.
- `DELETE /api/session/{session_id}`: forget a conversation. The last 6 turns are kept in memory and lost when the server restarts.
- `GET /api/health`: shows which components loaded and the error for any that failed. The server still starts when one model fails; that model's endpoints then return 503.

Error codes: `422` for bad input (empty text, no signer visible, unreadable video), `413` for a file that is too large, `415` for a file type that is not a video, `503` for a model that is not loaded.

### Pose file (`pose_url`)

One JSON per reply, 25 fps, in SMPL-X convention:

```json
{
  "format": "smplx-v1", "fps": 25, "frames": 76, "sentence": "...",
  "smplx": {
    "global_orient": [[3] x T], "body_pose": [[63] x T],        // 21 body joints, axis-angle (radians)
    "left_hand_pose": [[45] x T], "right_hand_pose": [[45] x T], // 15 joints each, flat_hand_mean = true
    "jaw_pose": [[3] x T], "leye_pose": [[3] x T], "reye_pose": [[3] x T],
    "expression": [[50] x T],                                   // FLAME-2020 expression coefficients
    "betas": [[10] x T], "transl": [[3] x T]
  },
  "joints": [[[x, y, z] x 127] x T],   // SMPL-X joint positions in metres, for a stick figure or retargeting
  "parents": [127]                     // kinematic tree for `joints`
}
```

- **SMPL-X avatar** (for example the SMPL-X Blender or Unity add-on, or three.js with an SMPL-X glTF): apply `smplx` directly, frame by frame.
- **Other rigs** (Mixamo, VRM, ReadyPlayerMe): retarget the rotations by joint name (SMPL-X order: pelvis, left_hip, right_hip, spine1, …, left_wrist, right_wrist, then the fingers). Or drive IK targets from `joints`.
- **Quickest option**: play `video_url`.

## Layout

```
signchat/
  config.py      defaults < YAML (SIGNCHAT_CONFIG) < env (SIGNCHAT_MOCK, SIGNCHAT_DIALOGUE)
  sign2text.py   video -> rtmlib keypoints -> Uni-Sign -> English (uses unisign/extract_pose, spec, dataset, model_adapter)
  text2sign.py   English -> SignSparK hand/body/face -> SMPL-X params + joints (+ mp4) (uses auslan_smplx/signspark_*)
  dialogue.py    reply generation backends + clean_reply (short, plain, signable)
  pipeline.py    one chat turn; in-memory sessions
  server.py      FastAPI app
  mock.py        fake models with the same outputs, for frontend work and tests
  _imports.py    keeps the Uni-Sign and SignSparK repos' same-named modules apart in one process
scripts/demo_cli.py   terminal chat
colab_server.ipynb    GPU server + public URL on Colab
```

## Notes for whoever runs it

- **Preprocessing matches training exactly.** Keypoints come from rtmlib `Wholebody` in lightweight mode with the largest person per frame. Videos are resampled to 25 fps and strided to at most 256 frames. Decoding is plain beam search with 4 beams. Text→sign runs 20 Euler steps, guidance 2.5, TF-IDF keyframe retrieval and σ=1 smoothing, the same as `colab_auslan_generate.ipynb`.
- **transformers version.** SignSparK needs `transformers==4.56.1`, while Uni-Sign was trained under 5.16.1 in Colab. Both models run in one process, so Uni-Sign runs under 4.56.1 here. mT5 loads the same way on both versions. Beam-search output has not yet been compared between them; re-scoring the val set once through `/api/translate/sign-to-text` would confirm it.
- **Memory.** Each SignSparK stream moves its 2.2 GB text encoder to the CPU after priming. The GPU holds the three generators, Uni-Sign (~2.4 GB) and Qwen-1.5B (~3 GB in bf16), which fits on a 16 GB T4. Requests share one lock for generation, so the server handles one sign generation at a time.
- **Startup cost.** About 13 GB is copied from Drive once per runtime. The three SignSparK generators are 0.83B-parameter UNets stored in fp32 (3.1 GB each), and every tensor is trained: `scripts/slim_assets.py` found nothing redundant in them. Only the 1.2 GB retrieval bank can be compacted, to about 0.15 GB (notebook section 2b, optional). Storing the generators in bf16 would halve them, but outputs would change slightly, so it would need validation first.
- **Using a big GPU.** On an A100 the notebook keeps the three M-CLIP text encoders on the GPU (`encoder_on_gpu`), runs pose extraction on the GPU in batches of 64, and uses Qwen2.5-7B for replies.
- **Speed and memory, measured on an A100 (2026-09-25; pose extraction on the GPU with cuDNN HEURISTIC).**

  | | time |
  |---|---|
  | text turn (5 turns) | 5.6 s mean, 6.6 s max: dialogue (Qwen-7B) 0.19 s + signing 5.4 s |
  | recognition (5 test clips, 2.3 s of video on average) | 1.34 s mean: pose extraction 1.00 s (max 1.7) + Uni-Sign 0.28 s + decoding 0.06 s |
  | signed turn | about 6.9 s: recognition + a text turn |

  GPU memory was 36.4 GB with Qwen-7B and the text encoders on the GPU. Estimated split: signing about 16 GB, Qwen-7B about 15 GB, recognition about 3 GB. System RAM was 8 GB.
  How pose extraction got from ~3.7 s to 1.0 s per clip:
  1. It had been running on the CPU without saying so. rtmlib's CPU `onnxruntime` shadows the GPU build, and onnxruntime-gpu 1.30 is a CUDA 13 build that cannot load next to CUDA 12 torch. Fixed by pinning onnxruntime-gpu 1.22.0 (CUDA 12); `/api/health` now reports `pose_providers`.
  2. onnxruntime's default cudnn_conv_algo_search, EXHAUSTIVE, picked slow algorithms for these models: 27 ms/frame for the pose model against 3.8 ms with HEURISTIC. `sign2text.pose_cudnn_algo` defaults to HEURISTIC. By `scripts/profile_pose.py`, confident keypoints move 0.02 px median (p99 2.5 px) and the chosen person never changes. By `scripts/check_pose_setting.py`, 4 of 5 test translations were identical. The fifth sits on a decision boundary: HEURISTIC gave the same text as CPU extraction, EXHAUSTIVE the odd one out.
  None of this fits a CPU web host such as Railway. Run this service on a GPU machine and have the web server call it over HTTP.
