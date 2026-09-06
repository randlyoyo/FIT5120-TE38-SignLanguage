# Deployment assets

Two files the recognition module needs are deliberately **not** in git. They are
build outputs, not source, and together they would grow a 2.3 MB repository to
roughly 57 MB for every clone, forever — git keeps blobs even after a later
delete.

| file | size | where it goes | how to get it |
|---|---|---|---|
| `models/bank.f32` | 39.5 MB | server only | `scripts/buildBank.mjs` (below) |
| `models/holistic_landmarker.task` | 13 MB | shipped to the browser | `scripts/fetchModel.sh` (below) |

`models/encoder.onnx` (3.7 MB) **is** committed — it is small enough, and it is
the one artefact with no reproducible download.

---

## `holistic_landmarker.task`

```sh
recognition/scripts/fetchModel.sh
```

Pinned to `float16/1`. Do **not** substitute a different bundle: the training
keypoints were extracted with this exact one, and a different version shifts the
landmark distribution in a way that shows up as unexplained accuracy loss rather
than an error. See `API.md` §2.

The script verifies the hash itself. The bundle used to extract the training
keypoints is:

```
sha256  e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8
```

For the client build, copy it into `website/client/public/models/` — Vite serves
`public/` verbatim, so it lands at `/models/holistic_landmarker.task`.

---

## `bank.f32` — the template bank

3215 words × 12 templates × 256 dims, float32, C order. A word's slice is at
byte offset `wordIndex * 12288`, length `12288`. `models/bank_index.json` holds
the word list and confirms those numbers.

It is produced from the training project, not from this repository:

```sh
node recognition/scripts/buildBank.mjs /path/to/signtest/export/bank.npy
```

The source `bank.npy` comes from `signtest/` — it is the embedding of all 38,580
training clips through `runs/final/encoder.pt`, grouped by word. Regenerating it
from scratch needs the keypoint dataset and the trained encoder, so in practice
you copy the exported file rather than rebuild it.

### Serving it

The API contract is `GET /api/recognition/template?word=<GLOSS>` returning 12 KB
(see `API.md` §6). Two workable shapes:

**Read the file at startup.** Simplest. 39.5 MB resident is fine on a Railway
dyno, and slicing is a `subarray` with no copy.

**Or seed it into MySQL** alongside the existing sign data, one row per gloss
with a 12 KB `BLOB`. This fits the project's existing `npm run seed` flow and
means the server holds no local state — worth it if the API is ever scaled to
more than one instance.

Either way the file belongs in deployment, not in the repo.
