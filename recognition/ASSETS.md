# Deployment assets

**Nothing needs fetching to deploy.** Everything the server requires is in the
repository, and the browser loads the MediaPipe bundle from Google's CDN.

This page records why, and what to do after retraining.

## What is committed

| file | size | used by |
|---|---|---|
| `models/encoder.onnx` | 3.7 MB | server |
| `models/bank.i8` | 9.9 MB | server |
| `models/manifest.json`, `models/bank_index.json` | 80 KB | server |

Vercel and Railway both deploy from git. An asset that is not committed is not
in the deployment, so a build-time fetch would be one more step to fail on two
platforms — worth avoiding for 14 MB.

The bank is stored int8 rather than the float32 the encoder emits: 9.9 MB
instead of 39.5 MB. The templates are unit vectors, so 127 levels per axis is
ample. Measured across three splits, quantising moved AUC by 0.0000, EER by at
most 0.01 points, and top-4 not at all. Zero values clipped.

## What is not committed, and why it does not matter

**`holistic_landmarker.task` (13 MB)** — the browser loads it directly from
Google:

```
https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task
```

Keep `/1/` in the path. `latest` would move the bundle under you, and a
different bundle shifts the landmark distribution in a way that costs accuracy
without raising an error. The expected hash is

```
sha256  e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8
```

`scripts/fetchModel.sh` downloads and hash-checks a local copy if you want one
for offline work. It is not needed for deployment.

**`bank.f32` (39.5 MB)** — the float32 intermediate. Only `bank.i8` ships.

## After retraining

Regenerate the bank from the training export and commit the result:

```sh
node recognition/scripts/buildBank.mjs /path/to/signtest/export/bank.npy
cd website/server && npm run test:recognition   # must still match Python
```

`bank.npy` comes from the training project: every clip in the Train split
embedded through `runs/final/encoder.pt` and grouped by word. It cannot be
rebuilt from this repository alone.

If the encoder itself changed, re-export `encoder.onnx` and regenerate
`test/golden.json` too — the conformance fixtures encode the model's outputs,
not just the feature pipeline.
