"""Dataset, normalisation statistics, conservative augmentation and collate.

Each preprocessed sample (<pose_dir>/<sample_id>.pt, see normalize_pose.py):
    pose (T, J, 2) signer space (NaN = joint never observed in the clip)
    mask (T, J) bool, text, length, norm, unisign_crop, mean_confidence

Batches:
    input_ids (B, L), attention_mask (B, L)      mT5 SentencePiece tokens of the English text
    pose (B, T, J, 2)                              NaN replaced by the training mean pose,
                                                   padded frames repeat the last real frame
    joint_valid (B, T, J) bool, frame_mask (B, T) bool, lengths (B,)
    camera (B, 8)                                  for the Uni-Sign converter
    text, uid                                      lists
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from slp_models import skeleton as sk
from slp_utils.unisign_format import camera_vector

# Left/right swap for the (non-default) horizontal flip.
_FLIP_PERM = list(range(sk.NUM_JOINTS))
for a, b in [(sk.L_EAR, sk.R_EAR), (sk.L_SHOULDER, sk.R_SHOULDER), (sk.L_ELBOW, sk.R_ELBOW),
             (sk.L_WRIST_BODY, sk.R_WRIST_BODY)]:
    _FLIP_PERM[a], _FLIP_PERM[b] = b, a
for i in range(21):
    _FLIP_PERM[sk.L_HAND_ROOT + i], _FLIP_PERM[sk.R_HAND_ROOT + i] = sk.R_HAND_ROOT + i, sk.L_HAND_ROOT + i
_bs = sk.PART_SLICES["brows"].start
for i in range(5):
    _FLIP_PERM[_bs + i], _FLIP_PERM[_bs + 5 + i] = _bs + 5 + i, _bs + i
_fs = sk.PART_SLICES["face_all"].start
for i in range(4):                      # jaw 0..8 mirrors around 4
    _FLIP_PERM[_fs + i], _FLIP_PERM[_fs + 8 - i] = _fs + 8 - i, _fs + i
for a, b in [(9, 13), (10, 12), (14, 16)]:  # inner lip = 68-point face 60..67
    _FLIP_PERM[_fs + a], _FLIP_PERM[_fs + b] = _fs + b, _fs + a


def read_split(path: Path) -> list[dict]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"split file not found: {path} (run preprocessing/make_splits.py)")
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "sample_id" not in rows[0]:
        raise KeyError(f"{path} has no sample_id column; regenerate it with make_splits.py")
    return rows


def load_sample(path: Path) -> dict | None:
    try:
        s = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[dataset] unreadable {path.name}: {exc}")
        return None
    pose, mask = s.get("pose"), s.get("mask")
    if not (isinstance(pose, torch.Tensor) and pose.dim() == 3 and pose.shape[1:] == (sk.NUM_JOINTS, sk.COORD_DIM)):
        print(f"[dataset] malformed pose in {path.name}: {getattr(pose, 'shape', None)}")
        return None
    if not (isinstance(mask, torch.Tensor) and mask.shape == pose.shape[:2]):
        print(f"[dataset] malformed mask in {path.name}")
        return None
    if s.get("layout") != sk.layout_metadata()["fingerprint"]:
        print(f"[dataset] {path.name}: layout fingerprint {s.get('layout')} != current; re-run normalize_pose.py")
        return None
    return s


def compute_norm_stats(rows: list[dict], pose_dir: Path, max_samples: int | None = None) -> dict:
    """Per-joint mean/std over VALID keypoints of the training split only."""
    J, C = sk.NUM_JOINTS, sk.COORD_DIM
    s1 = np.zeros((J, C)); s2 = np.zeros((J, C)); n = np.zeros((J, 1))
    conf_sum = np.zeros(J); conf_n = np.zeros(J)
    cams, lengths, lo, hi, src_fps = [], [], [], [], []
    for r in rows[: max_samples or len(rows)]:
        path = pose_dir / f"{r['sample_id']}.pt"
        if not path.is_file():
            continue  # rejected during preprocessing
        s = load_sample(path)
        if s is None:
            continue
        p = s["pose"].numpy().astype(np.float64)
        m = s["mask"].numpy()
        pv = np.where(m[..., None], np.nan_to_num(p), 0.0)
        s1 += pv.sum(0); s2 += (pv ** 2).sum(0); n += m.sum(0)[:, None]
        mc = s["mean_confidence"].numpy()
        seen = m.any(0)
        conf_sum += np.where(seen, mc, 0); conf_n += seen
        cams.append(camera_vector(s["norm"], s["unisign_crop"]))
        lengths.append(int(s["length"]))
        src_fps.append(float(s["norm"].get("source_fps", s.get("fps", 25.0))))
        vals = p[m]
        if len(vals):
            lo.append(np.percentile(vals, 1, axis=0)); hi.append(np.percentile(vals, 99, axis=0))
    if not lengths:
        raise RuntimeError("no readable training samples for normalisation statistics")
    mean = s1 / np.maximum(n, 1)
    var = s2 / np.maximum(n, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 0))
    never = (n[:, 0] == 0)
    mean[never] = 0.0
    std = np.maximum(std, 0.02)          # floor: finger joints have tiny variance
    cams = np.asarray(cams)
    lo, hi = np.min(lo, axis=0), np.max(hi, axis=0)
    return {
        "layout": sk.layout_metadata(),
        "n_samples": len(lengths),
        "mean": mean.tolist(), "std": std.tolist(),
        "mean_confidence": (conf_sum / np.maximum(conf_n, 1)).clip(0.31, 1.0).tolist(),
        "canonical_camera": np.median(cams, axis=0).tolist(),
        "source_fps_median": float(np.median(src_fps)),
        "view_bounds": {"xmin": float(lo[0]), "xmax": float(hi[0]), "ymin": float(lo[1]), "ymax": float(hi[1])},
        "length": {"median": float(np.median(lengths)), "p95": float(np.percentile(lengths, 95)),
                   "max": int(max(lengths))},
    }


def load_or_compute_stats(path: Path, rows: list[dict], pose_dir: Path, recompute: bool = False) -> dict:
    if path.is_file() and not recompute:
        stats = json.loads(path.read_text())
        if stats.get("layout", {}).get("fingerprint") == sk.layout_metadata()["fingerprint"]:
            return stats
        print("[stats] layout changed; recomputing")
    stats = compute_norm_stats(rows, pose_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=1))
    print(f"[stats] wrote {path} from {stats['n_samples']} training clips")
    return stats


class AuslanPoseDataset(Dataset):
    def __init__(self, rows: list[dict], pose_dir: Path, stats: dict, text_column: str = "text",
                 augment: dict | None = None, train: bool = False, cache: bool = True,
                 max_frames: int | None = None):
        self.pose_dir, self.stats = Path(pose_dir), stats
        self.text_column = text_column
        self.aug = augment or {}
        self.train = train
        self.cache = cache
        self.max_frames = max_frames
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self._mem: dict[str, dict] = {}
        self.rows = []
        missing = 0
        for r in rows:
            if (self.pose_dir / f"{r['sample_id']}.pt").is_file():
                self.rows.append(r)
            else:
                missing += 1
        if missing:
            print(f"[dataset] {missing}/{len(rows)} samples have no preprocessed pose (rejected or not run); skipped")
        if not self.rows:
            raise RuntimeError(f"no usable samples in {pose_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def _get(self, sid: str) -> dict | None:
        if sid in self._mem:
            return self._mem[sid]
        s = load_sample(self.pose_dir / f"{sid}.pt")
        if s is not None and self.cache:
            self._mem[sid] = s
        return s

    def _augment(self, pose: torch.Tensor, mask: torch.Tensor):
        a = self.aug
        rng = random
        # temporal scaling: resample to T * U(1 - r, 1 + r) frames
        r = float(a.get("temporal_scale", 0.0))
        if r > 0:
            T = pose.shape[0]
            newT = max(2, int(round(T * rng.uniform(1 - r, 1 + r))))
            u = torch.linspace(0, T - 1, newT)
            lo_i = u.floor().long().clamp(max=T - 1); hi_i = (lo_i + 1).clamp(max=T - 1)
            w = (u - lo_i.float())[:, None, None]
            pose = (1 - w) * pose[lo_i] + w * pose[hi_i]
            mask = mask[lo_i] & mask[hi_i]
        # frame dropout: replace a few interior frames by interpolation of neighbours
        p = float(a.get("frame_dropout", 0.0))
        if p > 0 and pose.shape[0] > 2:
            drop = (torch.rand(pose.shape[0]) < p)
            drop[0] = drop[-1] = False
            for t in torch.nonzero(drop).flatten().tolist():
                pose[t] = 0.5 * (pose[t - 1] + pose[t + 1])
        # global spatial scaling about the shoulder centre (origin)
        sc = float(a.get("spatial_scale", 0.0))
        if sc > 0:
            pose = pose * rng.uniform(1 - sc, 1 + sc)
        sig = float(a.get("coord_noise", 0.0))
        if sig > 0:
            pose = pose + torch.randn_like(pose) * sig
        if a.get("hflip", False) and rng.random() < float(a.get("hflip_prob", 0.5)):
            # Changes handedness. Only for data where dominance is not lexical.
            pose = pose[:, _FLIP_PERM].clone(); pose[..., 0] = -pose[..., 0]
            mask = mask[:, _FLIP_PERM]
        return pose, mask

    def __getitem__(self, i: int) -> dict | None:
        r = self.rows[i]
        s = self._get(r["sample_id"])
        if s is None:
            return None
        pose = s["pose"].clone()
        mask = s["mask"].clone()
        nan = torch.isnan(pose).any(-1)
        pose = torch.where(nan[..., None], self.mean.expand_as(pose), pose)
        mask &= ~nan
        if self.train and self.aug:
            pose, mask = self._augment(pose, mask)
        if self.max_frames and pose.shape[0] > self.max_frames:
            pose, mask = pose[: self.max_frames], mask[: self.max_frames]
        return {"uid": r["sample_id"], "text": r.get(self.text_column) or s.get("text", ""),
                "pose": pose, "mask": mask, "length": pose.shape[0],
                "camera": torch.tensor(camera_vector(s["norm"], s["unisign_crop"]), dtype=torch.float32)}


class Collator:
    def __init__(self, tokenizer, prompt: str = "", max_text_tokens: int = 64):
        self.tok = tokenizer
        self.prompt = prompt
        self.max_text_tokens = max_text_tokens

    def __call__(self, batch: list[dict | None]) -> dict | None:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        B, T = len(batch), max(b["length"] for b in batch)
        J, C = sk.NUM_JOINTS, sk.COORD_DIM
        pose = torch.zeros(B, T, J, C)
        jv = torch.zeros(B, T, J, dtype=torch.bool)
        fm = torch.zeros(B, T, dtype=torch.bool)
        for i, b in enumerate(batch):
            t = b["length"]
            pose[i, :t] = b["pose"]
            pose[i, t:] = b["pose"][-1]          # last-frame padding (never counted: fm=False)
            jv[i, :t] = b["mask"]
            fm[i, :t] = True
        texts = [self.prompt + b["text"] for b in batch]
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_text_tokens, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "pose": pose, "joint_valid": jv, "frame_mask": fm,
                "lengths": torch.tensor([b["length"] for b in batch]),
                "camera": torch.stack([b["camera"] for b in batch]),
                "text": [b["text"] for b in batch], "uid": [b["uid"] for b in batch]}


def build_tokenizer(mt5_dir: Path):
    from transformers import T5Tokenizer
    if not (Path(mt5_dir) / "spiece.model").is_file():
        raise FileNotFoundError(f"mT5 SentencePiece model not found in {mt5_dir}")
    # The same tokenizer (and legacy=False) Uni-Sign uses in models.py.
    return T5Tokenizer.from_pretrained(str(mt5_dir), legacy=False)
