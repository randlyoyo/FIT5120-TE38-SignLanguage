#!/usr/bin/env python3
"""Keypoint extraction. One script, both datasets, matching Uni-Sign exactly.

This mirrors Uni-Sign's own `demo/pose_extraction.py`:

  * rtmlib's `Wholebody`, NOT mmpose. rtmlib is the ONNX runtime the repo
    actually uses, with no mmcv/mmdet/mmpose dependency at all.
  * `mode="lightweight"` -- the default in both `demo/pose_extraction.py` and
    `demo/online_inference.py`, i.e. what the released pose data and their own
    inference path use. Under the hood that is RTMW-l-m at 192x256, plus
    YOLOX-tiny for detection. (The paper says "RTMPose-x"; the shipped code
    says otherwise, and the code is what produced the weights.)
  * Coordinates divided by [W, H] at extraction, exactly as upstream does.
    Everything after this point assumes frame-normalised input.

All 133 keypoints and all 133 scores are stored, not just the 69 that get used.
Upstream stores the full set and slices at load time, and following that means
the index selection lives in exactly one place (spec.py) and can be corrected
without re-extracting 70k clips.

Stored per clip (.npz):
    keypoints (T, 133, 2) float32, frame-normalised to [0,1]
    scores    (T, 133)    float32
    meta      json: spec fingerprint, model identity, frame size, fps

`--emit-pkl` additionally writes upstream's exact pickle format, so the same
extraction can be fed straight to the Uni-Sign repo's own scripts.

Batching (`--pose-batch`, default 32). rtmlib calls the pose model once per
person per frame. On a GPU that is dozens of tiny calls per second per process,
and running more worker processes only makes the card switch between more CUDA
contexts -- measured, going from 8 to 14 workers bought 23%. The pose model's
ONNX export has a dynamic batch dimension, so crops from many frames can go
through one call instead. The detector's export is fixed at batch 1 and still
runs per frame.

Nothing else changes: every crop is prepared and decoded by rtmlib's own
RTMPose.preprocess / RTMPose.postprocess, and rtmlib's "nobody detected -> use
the whole image" fallback is kept. `verify_batching.py` compares the two paths
element by element on real clips. `--pose-batch 1` is the upstream path exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402
from manifest import read_manifest  # noqa: E402

DEFAULT_MODE = "lightweight"
DEFAULT_BACKEND = "onnxruntime"
# Crops per pose-model call. Memory per call is batch x 3 x 256 x 192 float32
# (~19 MB at 32), bounded no matter how long the clip is.
DEFAULT_POSE_BATCH = 32
# See _select_person for the evidence behind this default.
DEFAULT_PERSON_SELECT = "largest"


class _VideoSource:
    """Opens a plain path or an `archive.zip::member` reference.

    Zipped members are staged to a temp file. MM-WLAuslan ships as multi-GB
    archives and unpacking all of them to read each clip once wastes an
    inconvenient amount of disk.
    """

    def __init__(self, ref: str, tmpdir: Path):
        self._tmp: Path | None = None
        if "::" in ref:
            archive, member = ref.split("::", 1)
            fd, tmp = tempfile.mkstemp(suffix=Path(member).suffix or ".mp4",
                                       dir=str(tmpdir))
            os.close(fd)
            self._tmp = Path(tmp)
            with zipfile.ZipFile(archive) as zf, zf.open(member) as src, \
                    self._tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            self.path = self._tmp
        else:
            self.path = Path(ref)
            if not self.path.exists():
                raise FileNotFoundError(ref)

    def close(self) -> None:
        if self._tmp is not None and self._tmp.exists():
            self._tmp.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _read_frames(path: Path, max_frames: int | None = None):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"no decodable frames in {path}")
    h, w = frames[0].shape[:2]
    return frames, w, h, fps


def _select_person(kps: np.ndarray, scores: np.ndarray, prev: np.ndarray | None,
                   policy: str, boxes: np.ndarray | None = None) -> int:
    """Which detection is the signer.

    'largest' picks the detection with the biggest box, every frame. Signer-Only
    crops are cut around the annotated signer, so the signer dominates the
    frame. This was measured against Auslan-Daily's official pose annotation,
    which marks the signer in every frame. On 49 held-out clips from video 46
    (3,587 frames, none of them looked at while choosing the rule), the largest
    box was the signer in 99.97% of frames, against 97.99% for 'track' and
    98.75% for upstream's 'first'. On the 38 high-risk development clips
    (2,439 frames), the figures were 99.96%, 88.97% and 89.13%.

    'track' does not fail frame by frame. It commits to detection 0 in the
    first frame and then follows that person, so when detection 0 is a
    bystander the whole clip is the wrong person. Three clips came out 0%
    correct that way. Any rule that tracks inherits this; per-frame 'largest'
    has no memory to lock in a mistake.

    'largest' does not fix a detector box that swallows two people. There the
    biggest box is the merged one, and the pose model fits a single skeleton
    across both bodies. No selection rule can recover those frames, because
    none of the candidates is the signer.

    Upstream takes detection 0 unconditionally, which is fine for their corpora
    -- single signer, centred. Auslan-Daily's Communication subset is
    in-the-wild dialogue that can have more than one person in frame, and there
    "detection 0" can refer to a different person from one frame to the next,
    producing a track that is a blend of two signers. `track` keeps the choice
    continuous by following the nearest torso.

    This is the one deliberate deviation from upstream. It changes which person
    is measured, never how a pose is represented, and the policy is recorded in
    every clip's metadata so a set extracted under two policies is detectable.
    """
    if kps.shape[0] == 0:
        return -1
    if policy == "largest":
        if boxes is None or len(boxes) != kps.shape[0]:
            raise ValueError("'largest' needs one detector box per candidate")
        b = np.asarray(boxes, dtype=np.float64)
        return int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    if policy == "first" or kps.shape[0] == 1 or prev is None:
        return 0
    body = spec.PART_INDICES["body"]
    prev_c = prev[body].mean(axis=0)
    d = [float(np.linalg.norm(k[body].mean(axis=0) - prev_c)) for k in kps]
    return int(np.argmin(d))


class PoseExtractor:
    def __init__(self, mode: str = DEFAULT_MODE, backend: str = DEFAULT_BACKEND,
                 device: str = "cpu", person_select: str = DEFAULT_PERSON_SELECT,
                 pose_batch: int = DEFAULT_POSE_BATCH):
        try:
            from rtmlib import Wholebody
        except ImportError as exc:
            raise SystemExit(
                "rtmlib is not installed. This is the estimator Uni-Sign "
                "actually uses -- do NOT substitute MediaPipe or mmpose, the "
                "pre-trained weights assume this keypoint layout.\n"
                "  pip install rtmlib onnxruntime"
            ) from exc
        if device == "mps" and backend == "onnxruntime":
            raise SystemExit(
                "device=mps is not usable here. onnxruntime routes it to the "
                "CoreML execution provider, and YOLOX's detection output has a "
                "dynamic shape that CoreML cannot handle when a frame contains "
                "no person:\n"
                "  Input (1219) has a dynamic shape ({-1}) but the runtime "
                "shape ({0}) has zero elements\n"
                "It does not fail on the first frame -- it fails on the first "
                "frame where nobody is detected, which in a 70k-clip run means "
                "hours in. Use --device cpu, or cuda on a GPU box."
            )

        self.mode = mode
        self.backend = backend
        self.device = device
        self.person_select = person_select
        self.pose_batch = max(1, int(pose_batch))
        # rtmlib exposes no __version__; take it from the installed
        # distribution so the extraction is pinned to a build, not just a name.
        try:
            from importlib.metadata import version
            self.rtmlib_version = version("rtmlib")
        except Exception:
            self.rtmlib_version = "unknown"
        self.model_tag = f"rtmlib.Wholebody/{mode}"
        self._wholebody = Wholebody(to_openpose=False, mode=mode,
                                    backend=backend, device=device)

    def _candidates_per_frame(self, frames):
        """Upstream path: detector then pose model, one frame and one crop at a time.

        This is Wholebody.__call__ unrolled -- `bboxes = det(img)` then
        `pose(img, bboxes)` -- so the detector's boxes stay available to person
        selection. With no detection the whole image is passed as the box,
        which is exactly what RTMPose does itself when given an empty list.
        """
        det = self._wholebody.det_model
        pose = self._wholebody.pose_model
        for frame in frames:
            img = np.uint8(frame)
            bboxes = det(img)
            if len(bboxes) == 0:
                bboxes = [[0, 0, img.shape[1], img.shape[0]]]
            kps, scores = pose(img, bboxes=bboxes)
            yield kps, scores, np.asarray(bboxes, dtype=np.float32)

    def _candidates_batched(self, frames):
        """The same computation, with many crops per pose-model call.

        Returns, per frame, (keypoints (n,133,2), scores (n,133)) for every
        detected person in detection order -- exactly what Wholebody returns --
        so person selection downstream sees identical input either way.
        """
        det = self._wholebody.det_model
        pose = self._wholebody.pose_model
        if getattr(pose, "to_openpose", False):
            raise RuntimeError("batched path assumes to_openpose=False")
        sess = pose.session
        in_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]

        results = [[] for _ in frames]   # per frame: [(kpts, scores), ...]
        frame_boxes = [None] * len(frames)
        pending = []                     # (frame index, center, scale, crop)

        def flush():
            if not pending:
                return
            # BaseTool.inference, stacked: HWC -> CHW, float32, batch on axis 0.
            x = np.stack([np.ascontiguousarray(c.transpose(2, 0, 1),
                                               dtype=np.float32)
                          for _, _, _, c in pending])
            outs = sess.run(out_names, {in_name: x})
            for j, (t, center, scale, _) in enumerate(pending):
                k, sc = pose.postprocess([o[j:j + 1] for o in outs],
                                         center, scale)
                results[t].append((k, sc))
            pending.clear()

        for t, frame in enumerate(frames):
            img = np.uint8(frame)
            bboxes = det(img)
            # RTMPose.__call__ never returns zero people: with no detection it
            # falls back to the whole image as the box. Reproduce that, or every
            # frame where the detector misses comes out different.
            if len(bboxes) == 0:
                bboxes = [[0, 0, img.shape[1], img.shape[0]]]
            frame_boxes[t] = np.asarray(bboxes, dtype=np.float32)
            for bbox in bboxes:
                crop, center, scale = pose.preprocess(img, bbox)
                pending.append((t, center, scale, crop))
                if len(pending) >= self.pose_batch:
                    flush()
        flush()

        for t, per_frame in enumerate(results):
            yield (np.concatenate([k for k, _ in per_frame], axis=0),
                   np.concatenate([s for _, s in per_frame], axis=0),
                   frame_boxes[t])

    def run(self, frames, width: int, height: int):
        source = (self._candidates_batched(frames) if self.pose_batch > 1
                  else self._candidates_per_frame(frames))
        return self._assemble(source, len(frames), width, height)

    def _assemble(self, source, T: int, width: int, height: int):
        """Per-frame candidates -> one signer's (T,133,2) keypoints, (T,133) scores.

        Shared by both inference paths, so person selection and normalisation
        cannot drift between them. Also returns an audit trail -- how many people
        each frame had and which box was chosen -- so a clip's selection can be
        re-examined later without running inference again.
        """
        kp_out = np.zeros((T, spec.NUM_WHOLEBODY_KEYPOINTS, 2), dtype=np.float32)
        sc_out = np.zeros((T, spec.NUM_WHOLEBODY_KEYPOINTS), dtype=np.float32)
        n_people = np.zeros(T, dtype=np.int16)
        chosen_box = np.zeros((T, 4), dtype=np.float32)
        wh = np.array([width, height], dtype=np.float32)

        prev = None
        for t, (kps, scores, boxes) in enumerate(source):
            kps = np.asarray(kps, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
            if kps.ndim == 2:               # a single detection, unbatched
                kps, scores = kps[None], scores[None]
            n_people[t] = kps.shape[0]
            i = _select_person(kps, scores, prev, self.person_select, boxes=boxes)
            if i < 0:
                continue                    # no detection: leave zeros
            if kps.shape[1] != spec.NUM_WHOLEBODY_KEYPOINTS:
                raise RuntimeError(
                    f"{self.model_tag} returned {kps.shape[1]} keypoints, "
                    f"expected {spec.NUM_WHOLEBODY_KEYPOINTS} (COCO-WholeBody)."
                )
            prev = kps[i]
            kp_out[t] = kps[i] / wh          # upstream: keypoints / [W, H]
            sc_out[t] = scores[i]
            chosen_box[t] = np.asarray(boxes[i], dtype=np.float32) / np.tile(wh, 2)
        return kp_out, sc_out, {"n_people": n_people, "chosen_box": chosen_box}


def _matches_current(path: Path, ex: "PoseExtractor") -> bool:
    """Was this existing .npz made under the same spec and person-selection policy?

    Skipping any existing file would quietly keep clips from an earlier run
    under an older policy next to new ones -- a mixed set that only
    verify_pose's consistency check would catch, after the fact. Unreadable
    files count as stale.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            m = json.loads(str(z["meta"]))
    except Exception:
        return False
    return (m.get("schema_fingerprint") == spec.SCHEMA_FINGERPRINT
            and m.get("person_select") == ex.person_select)


def _meta(row: dict, ex: PoseExtractor, width: int, height: int, fps: float,
          n_frames: int) -> dict:
    return {
        "uid": row["uid"], "dataset": row["dataset"], "subset": row["subset"],
        "split": row["split"], "text": row["text"], "gloss": row.get("gloss"),
        "camera": row.get("camera"), "source": row["video"],
        "spec_version": spec.SPEC_VERSION,
        "schema_fingerprint": spec.SCHEMA_FINGERPRINT,
        "pose_model": ex.model_tag, "pose_backend": ex.backend,
        "rtmlib_version": ex.rtmlib_version,
        "person_select": ex.person_select,
        # Recorded, but not one of verify_pose's identity fields: the batched
        # and per-frame paths are verified equivalent, so a set that mixes them
        # (clips extracted before and after batching existed) is consistent.
        "pose_batch": ex.pose_batch,
        "coordinate_space": "frame_normalised",
        "width": int(width), "height": int(height), "fps": float(fps),
        "n_frames": int(n_frames),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", default=DEFAULT_MODE,
                    choices=["performance", "lightweight", "balanced"])
    ap.add_argument("--backend", default=DEFAULT_BACKEND,
                    choices=["opencv", "onnxruntime", "openvino"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--person-select", default=DEFAULT_PERSON_SELECT,
                    choices=["track", "first", "largest"],
                    help="'first' reproduces upstream exactly; 'track' follows "
                         "detection 0 from the first frame; 'largest' takes the "
                         "biggest box each frame (see _select_person)")
    ap.add_argument("--pose-batch", type=int, default=DEFAULT_POSE_BATCH,
                    help="crops per pose-model call; 1 = upstream per-frame "
                         "path exactly")
    ap.add_argument("--emit-pkl", action="store_true",
                    help="also write upstream's pickle format")
    ap.add_argument("--limit", type=int, default=None,
                    help="only consider the first N manifest rows")
    ap.add_argument("--max-new", type=int, default=None,
                    help="stop after N clips are NEWLY extracted, ignoring ones "
                         "already present. Unlike --limit this makes repeated "
                         "runs advance, which is what checkpointing to Drive "
                         "from a Colab session that can be cut off needs.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    rows = read_manifest(args.manifest)
    if args.num_shards > 1:
        rows = rows[args.shard :: args.num_shards]
    if args.limit:
        rows = rows[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    ex = PoseExtractor(args.mode, args.backend, args.device, args.person_select,
                       args.pose_batch)
    print(f"model={ex.model_tag} backend={ex.backend} device={ex.device} "
          f"person={ex.person_select} pose_batch={ex.pose_batch} "
          f"spec={spec.SCHEMA_FINGERPRINT}",
          file=sys.stderr)

    tmpdir = Path(tempfile.mkdtemp(prefix="unisign-extract-"))
    done = skipped = failed = stale = 0
    t0 = time.time()
    try:
        for i, row in enumerate(rows, 1):
            dest = args.out / f"{row['uid']}.npz"
            if dest.exists() and not args.overwrite:
                if _matches_current(dest, ex):
                    skipped += 1
                    continue
                stale += 1              # made under another spec/policy: redo it
            try:
                with _VideoSource(row["video"], tmpdir) as src:
                    frames, w, h, fps = _read_frames(src.path)
                    kp, sc, audit = ex.run(frames, w, h)
                meta = _meta(row, ex, w, h, fps, len(frames))
                # Write through a file handle, not a path: np.savez_compressed
                # appends ".npz" to any path that lacks it, so a ".npz.part"
                # temp name silently becomes ".npz.part.npz" and the rename
                # below fails.
                tmp_out = dest.with_name(dest.name + ".part")
                with tmp_out.open("wb") as fh:
                    np.savez_compressed(fh, keypoints=kp, scores=sc,
                                        meta=json.dumps(meta), **audit)
                tmp_out.replace(dest)   # atomic: a killed job leaves no half file
                if args.emit_pkl:
                    with (args.out / f"{row['uid']}.pkl").open("wb") as fh:
                        pickle.dump({"keypoints": [k[None] for k in kp],
                                     "scores": [s[None] for s in sc]}, fh)
                done += 1
            except Exception as exc:
                failed += 1
                print(f"FAIL {row['uid']}: {exc}", file=sys.stderr)
            if args.max_new and done >= args.max_new:
                print(f"reached --max-new {args.max_new}", file=sys.stderr)
                break
            if i % 100 == 0:
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(rows)}  {rate:.2f} clip/s  done={done} "
                      f"skip={skipped} fail={failed}", file=sys.stderr)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"extracted={done} skipped={skipped} failed={failed}"
          + (f" (of which re-extracted because an earlier run used a different "
             f"spec or person-selection policy: {stale})" if stale else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
