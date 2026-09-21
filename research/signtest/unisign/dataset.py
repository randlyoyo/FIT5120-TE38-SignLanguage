#!/usr/bin/env python3
"""Datasets and batching. Both corpora go through one class.

Output matches what Uni-Sign's `Uni_Sign.forward` consumes, because it is a
port of its collate_fn (datasets.py) rather than a reimplementation:

    src_input['body']        (B, T,  9, 3)  float32
    src_input['left']        (B, T, 21, 3)
    src_input['right']       (B, T, 21, 3)
    src_input['face_all']    (B, T, 18, 3)
    src_input['attention_mask'] (B, T) long, 1 for real frames
    tgt_input['gt_sentence'] list[str]

Two details are upstream's and are easy to get wrong by "improving" them:

  Padding repeats the LAST FRAME, it does not zero-pad. A zero frame is not a
  neutral value here -- after normalisation, zero means "joint below the
  confidence threshold", so zero padding reads as a signer who vanishes rather
  than one who stopped.

  Confidence is the third channel and is already folded in by
  spec.load_part_kp, which zeroes sub-threshold joints. There is no separate
  mask to maintain, and adding one would put the same information in two places
  that can disagree.

Gloss targets. MM-WLAuslan glosses carry disambiguation in brackets --
"SHAVE (UNDERARM)", "YOUR (PLURAL)". Those brackets are lexicographic notation,
not English. Feeding them as SLT targets teaches the decoder to emit bracketed
citation forms, which then leak into Auslan-Daily sentence output. The default
mode strips them; `raw` keeps them if you want to measure that effect.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402
from manifest import read_manifest  # noqa: E402

_PAREN = re.compile(r"\s*\([^)]*\)")


def gloss_to_text(gloss: str, mode: str = "strip_paren") -> str:
    if mode == "raw":
        return gloss
    text = gloss
    if mode in ("strip_paren", "strip_paren_lower"):
        text = _PAREN.sub("", text).strip()
    if mode in ("lower", "strip_paren_lower"):
        text = text.lower()
    return text or gloss


def target_text_for(row: dict, gloss_text_mode: str = "strip_paren") -> str:
    """The string the SLT head is trained to produce, for one manifest row.

    Module level, not a method, so the vocabulary of a whole corpus can be
    computed before any single stage's dataset exists.
    """
    if row["dataset"] == "mmwlauslan" and row.get("gloss"):
        return gloss_to_text(row["gloss"], gloss_text_mode)
    return row["text"]


def _subsample(T: int, max_length: int, deterministic: bool) -> np.ndarray:
    """Upstream picks `sorted(random.sample(range(T), max_length))`.

    That is kept for training -- it is a mild augmentation and matches what the
    weights saw. For validation and test it is replaced by a uniform stride:
    a random subsample makes the score depend on the seed, and a metric that
    moves when you re-run it cannot be used to compare arms.
    """
    if T <= max_length:
        return np.arange(T)
    if deterministic:
        return np.linspace(0, T - 1, max_length).round().astype(np.int64)
    return np.asarray(sorted(random.sample(range(T), k=max_length)), dtype=np.int64)


class SignPoseDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        npz_dir: str | Path,
        *,
        datasets: list[str] | None = None,
        subsets: list[str] | None = None,
        splits: list[str] | None = None,
        max_length: int = 256,          # utils.get_args_parser default
        gloss_text_mode: str = "strip_paren",
        deterministic: bool = False,
        require_npz: bool = True,
        strict_fingerprint: bool = True,
        exclude: frozenset[str] = frozenset(),
    ):
        self.npz_dir = Path(npz_dir)
        self.max_length = max_length
        self.gloss_text_mode = gloss_text_mode
        self.deterministic = deterministic
        self.strict_fingerprint = strict_fingerprint

        rows = read_manifest(Path(manifest))
        if datasets:
            rows = [r for r in rows if r["dataset"] in datasets]
        if subsets:
            rows = [r for r in rows if r["subset"] in subsets]
        if splits:
            rows = [r for r in rows if r["split"] in splits]
        # Clips verify_pose.py found unusable (mirrored). Dropped before the
        # missing-.npz check, so a listed clip need not have been extracted.
        n = len(rows)
        rows = [r for r in rows if r["uid"] not in exclude]
        self.excluded = n - len(rows)

        kept, missing = [], 0
        for r in rows:
            if (self.npz_dir / f"{r['uid']}.npz").exists():
                kept.append(r)
            else:
                missing += 1
        if missing and require_npz:
            raise FileNotFoundError(
                f"{missing}/{len(rows)} manifest rows have no .npz in {npz_dir}. "
                "Finish the extraction rather than training on whatever is "
                "present, which silently changes the corpus between runs."
            )
        self.rows = kept
        self.missing = missing
        if not self.rows:
            raise ValueError("dataset is empty after filtering")

    def __len__(self) -> int:
        return len(self.rows)

    def target_text(self, row: dict) -> str:
        return target_text_for(row, self.gloss_text_mode)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        with np.load(self.npz_dir / f"{row['uid']}.npz", allow_pickle=False) as z:
            kp, sc = z["keypoints"], z["scores"]
            meta = json.loads(str(z["meta"]))

        if self.strict_fingerprint and meta.get("schema_fingerprint") != spec.SCHEMA_FINGERPRINT:
            raise RuntimeError(
                f"{row['uid']}: extracted under fingerprint "
                f"{meta.get('schema_fingerprint')}, spec.py is now "
                f"{spec.SCHEMA_FINGERPRINT}. Re-extract; do not train across "
                "two representations."
            )

        keep = _subsample(kp.shape[0], self.max_length, self.deterministic)
        parts = spec.load_part_kp(kp[keep], sc[keep])

        out = {
            "uid": row["uid"], "dataset": row["dataset"], "subset": row["subset"],
            "text": self.target_text(row), "length": int(len(keep)),
        }
        for part in spec.PART_ORDER:
            out[part] = torch.from_numpy(parts[part])
        return out


def collate(batch: list[dict]) -> dict:
    """Port of Uni-Sign's collate_fn: last-frame padding + a frame mask."""
    B, T = len(batch), max(int(b["length"]) for b in batch)
    src: dict = {}
    for part in spec.PART_ORDER:
        J = spec.PART_SIZES[part]
        buf = torch.zeros(B, T, J, 3, dtype=torch.float32)
        for i, b in enumerate(batch):
            t = int(b["length"])
            buf[i, :t] = b[part]
            if t < T:
                buf[i, t:] = b[part][-1][None].expand(T - t, -1, -1)
        src[part] = buf

    mask = torch.zeros(B, T, dtype=torch.long)
    for i, b in enumerate(batch):
        mask[i, : int(b["length"])] = 1
    src["attention_mask"] = mask
    src["src_length_batch"] = torch.tensor([int(b["length"]) for b in batch])
    src["name_batch"] = [b["uid"] for b in batch]

    src["uid"] = [b["uid"] for b in batch]
    src["dataset"] = [b["dataset"] for b in batch]
    src["subset"] = [b["subset"] for b in batch]
    src["text"] = [b["text"] for b in batch]
    return src


# --- vocabulary / OOV -------------------------------------------------------
#
# The OOV rate decides whether MM-WLAuslan's vocabulary adaptation did anything
# and whether Arm D is worth building, so it is a reported number, not a
# diagnostic. Computed over the English target text, since that is what the
# model has to produce.

_WORD = re.compile(r"[a-z']+")


def tokenise(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def vocabulary(texts: list[str]) -> Counter:
    v: Counter = Counter()
    for t in texts:
        v.update(tokenise(t))
    return v


def oov_report(train_texts: list[str], test_texts: list[str]) -> dict:
    train_v = vocabulary(train_texts)
    test_v = vocabulary(test_texts)
    oov_types = [w for w in test_v if w not in train_v]
    oov_tokens = sum(test_v[w] for w in oov_types)
    total = sum(test_v.values())
    return {
        "train_types": len(train_v), "test_types": len(test_v),
        "oov_types": len(oov_types),
        "oov_type_rate": len(oov_types) / max(len(test_v), 1),
        "oov_token_rate": oov_tokens / max(total, 1),
        "examples": sorted(oov_types, key=lambda w: -test_v[w])[:25],
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="inspect a manifest + npz set")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--subsets", nargs="*")
    ap.add_argument("--splits", nargs="*")
    a = ap.parse_args()

    ds = SignPoseDataset(a.manifest, a.npz_dir, datasets=a.datasets,
                         subsets=a.subsets, splits=a.splits, require_npz=False)
    print(f"{len(ds)} clips ({ds.missing} manifest rows had no .npz)")
    b = collate([ds[i] for i in range(min(4, len(ds)))])
    for part in spec.PART_ORDER:
        print(f"  {part:<9} {tuple(b[part].shape)}")
    print(f"  attention_mask {tuple(b['attention_mask'].shape)}")
    print(f"  texts {b['text']}")
