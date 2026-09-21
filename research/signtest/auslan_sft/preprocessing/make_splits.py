#!/usr/bin/env python3
"""Reproducible, sample-level train/val/test split of annotations.csv.

Leakage rules enforced here:
  * the unit of splitting is the video; duplicated video paths are collapsed
    (first occurrence kept) so one clip can never sit in two splits;
  * with `split.group_column` set (e.g. a signer id, or the text itself), all
    rows sharing that value go to the same split -- use it when the same
    sentence is recorded many times, otherwise the test set measures recall of
    training sentences, not generalisation;
  * with `split.method: column`, an existing split column is respected
    (e.g. Auslan-Daily's official split). Use this whenever an evaluator (SLR
    back-translation model) was trained on that dataset's official train split,
    or back-translation scores on "test" will be inflated by leakage.

The split is a pure function of (rows, seed, ratios, grouping): re-running it
gives byte-identical files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slp_utils.config import load_config, resolve_path  # noqa: E402

SPLITS = ("train", "val", "test")
_SPLIT_ALIASES = {"train": "train", "dev": "val", "val": "val", "valid": "val",
                  "validation": "val", "test": "test"}


def read_annotations(path: Path, dcfg: dict) -> list[dict]:
    vcol = dcfg.get("video_column", "video_path")
    tcol = dcfg.get("text_column", "text")
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        for need in (vcol, tcol):
            if need not in cols:
                raise KeyError(f"{path}: column {need!r} not found; columns are {cols}. "
                               "Set data.video_column / data.text_column in the config.")
        rows, bad = [], 0
        for r in reader:
            if not (r.get(vcol) or "").strip() or not (r.get(tcol) or "").strip():
                bad += 1
                continue
            rows.append(r)
    if bad:
        print(f"[annotations] skipped {bad} rows with an empty video path or text")
    return rows


def sample_id(row: dict, dcfg: dict) -> str:
    """Stable file-name-safe id: the id column if configured, else the video stem."""
    idcol = dcfg.get("id_column")
    if idcol and row.get(idcol):
        raw = row[idcol]
    else:
        raw = Path(row[dcfg.get("video_column", "video_path")]).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    if not safe:
        safe = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return safe


def normalise_text(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", t.lower())).strip()


def make_splits(rows: list[dict], dcfg: dict, scfg: dict) -> dict[str, list[dict]]:
    vcol = dcfg.get("video_column", "video_path")
    seen, uniq, dups = set(), [], 0
    for r in rows:
        key = r[vcol].strip()
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        uniq.append(r)
    if dups:
        print(f"[split] dropped {dups} duplicate video rows")

    ids = Counter(sample_id(r, dcfg) for r in uniq)
    clash = [k for k, n in ids.items() if n > 1]
    if clash:
        raise ValueError(f"{len(clash)} different videos map to the same sample id "
                         f"(e.g. {clash[:3]}); set data.id_column to a unique column")

    method = scfg.get("method", "random")
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    if method == "column":
        col = scfg.get("split_column", "split")
        for r in uniq:
            s = _SPLIT_ALIASES.get(str(r.get(col, "")).strip().lower())
            if s is None:
                raise ValueError(f"row {sample_id(r, dcfg)}: split value {r.get(col)!r} "
                                 f"not in {sorted(_SPLIT_ALIASES)}")
            out[s].append(r)
        return out
    if method != "random":
        raise ValueError(f"split.method must be random|column, got {method!r}")

    ratios = [float(scfg.get("train_ratio", 0.8)), float(scfg.get("val_ratio", 0.1)),
              float(scfg.get("test_ratio", 0.1))]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    gcol = scfg.get("group_column")
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in uniq:
        if gcol == "text":
            g = normalise_text(r[dcfg.get("text_column", "text")])
        elif gcol:
            if gcol not in r:
                raise KeyError(f"split.group_column {gcol!r} not in annotations")
            g = str(r[gcol])
        else:
            g = sample_id(r, dcfg)
        groups[g].append(r)
    keys = sorted(groups)
    random.Random(int(scfg.get("seed", 42))).shuffle(keys)
    n = len(uniq)
    bounds = [ratios[0] * n, (ratios[0] + ratios[1]) * n]
    count = 0
    for k in keys:
        s = "train" if count < bounds[0] else ("val" if count < bounds[1] else "test")
        out[s].extend(groups[k])
        count += len(groups[k])
    for s in SPLITS:
        out[s].sort(key=lambda r: sample_id(r, dcfg))
    return out


def leakage_report(splits: dict[str, list[dict]], dcfg: dict) -> None:
    tcol = dcfg.get("text_column", "text")
    train_text = {normalise_text(r[tcol]) for r in splits["train"]}
    for s in ("val", "test"):
        if not splits[s]:
            continue
        overlap = sum(normalise_text(r[tcol]) in train_text for r in splits[s])
        print(f"[split] {s}: {len(splits[s])} clips, {overlap} ({overlap/len(splits[s]):.0%}) "
              "have a sentence that also occurs in train"
              + ("  <- consider split.group_column: text" if overlap else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="make train/val/test CSVs")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    d, scfg = cfg["data"], cfg.get("split", {})
    rows = read_annotations(resolve_path(cfg, d["annotations_csv"]), d)
    splits = make_splits(rows, d, scfg)
    out_dir = resolve_path(cfg, d["split_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) + ["sample_id"] if rows else ["sample_id"]
    for s in SPLITS:
        with (out_dir / f"{s}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in splits[s]:
                w.writerow({**r, "sample_id": sample_id(r, d)})
        with (out_dir / f"{s}.txt").open("w", encoding="utf-8") as fh:
            fh.writelines(sample_id(r, d) + "\n" for r in splits[s])
        print(f"[split] wrote {out_dir / (s + '.csv')} ({len(splits[s])} rows)")
    leakage_report(splits, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
