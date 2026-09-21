#!/usr/bin/env python3
"""Convert a downloaded Auslan-Daily release into the records JSON manifest.py reads.

What the release actually is (uq-cvlab.github.io/Auslan-Daily-Dataset, and the
public Drive folder linked from its Dataset Download page):

    Auslan_Daily/
      Auslan-Daily Communication/
      Auslan-Daily News/
      Dataset Split Table

Each sub-dataset ships several video variants, described on the Dataset Format
page:

    Signer Only Video Clips   cropped to the signer using the ground-truth
                              bounding box
    Multi-Person Video Clips  the same clips, uncropped
    Whole Video               the full episode
    Pose Annotation           pose sequences for every person in a clip

**Use the Signer Only clips.** Auslan-Daily has one to ten people in frame and
the paper counts multi-person gesture interference as part of the challenge;
the cropped variant removes that problem at the source, which is strictly
better than asking a pose estimator to guess who is signing. The tracking
heuristic in extract_pose.py is then a safety net rather than the mechanism.

The bundled Pose Annotation is the dataset's own format, not Uni-Sign's
keypoint layout, so it is not a shortcut past extraction -- extract with
extract_pose.py like everything else, or the two corpora are not comparable.

Licence: CC BY 4.0. Commercial use is permitted with attribution, which is
worth recording since it is the one dataset question with a business answer.

The split tables are XLSX. Their columns, per the release's own
ReadMe_Sign_Language_Translation.txt:

    Video_Name       the whole episode the clip came from
    Video_Clip_Name  the clip name, as used by the Multi-Person folder
    Subtitle         the English translation -- the SLT target
    Split            train | dev | test
    Signer_ID        which signer
    Signer_Video     News only: the SIGNER-ONLY clip name

That last row is a trap, and it bites in both directions. The Signer-Only
files are always named `<clip>_signer.mp4`. For News that full name sits in
`Signer_Video`, so keying on `Video_Clip_Name` matches nothing. For
Communication there is no `Signer_Video` column at all, so the `_signer`
suffix has to be added. Both were verified against the shipped archives:
`Signer/video_1_35_signer.mp4` for Communication, and the table row for that
clip says `video_1_35`.

So lookup tries the id as given, then the id with `_signer` appended, and
reports which rule matched how often.

Column matching is still done by meaning rather than by position, so a
re-released table with renamed headings is reported rather than silently
mis-parsed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# The Signer-Only crops carry this suffix; the split tables mostly do not.
SIGNER_SUFFIX = "_signer"

# Column-name candidates, most specific first. Matched case-insensitively
# against normalised headings (non-alphanumerics stripped).
COLUMN_ALIASES: dict[str, list[str]] = {
    # Signer_Video first: where it exists (News) it is the Signer-Only clip.
    "id": ["signervideo", "videoclipname", "clipname", "clipid", "videoid",
           "clip", "filename", "file", "id"],
    "text": ["subtitle", "sentence", "translation", "text", "englishsentence",
             "english", "caption", "annotation"],
    "split": ["split", "set", "partition", "phase", "datasplit"],
}

SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val", "dev": "val",
    "test": "test", "testing": "test", "eval": "test",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _read_table(path: Path) -> list[dict]:
    """Read a split table. CSV/TSV/JSON only -- export XLSX to CSV first."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise TypeError(f"{path}: expected a JSON list of rows")
        return data
    if suffix in {".xlsx", ".xlsm"}:
        try:
            import openpyxl
        except ImportError:
            raise SystemExit(
                "the release ships XLSX split tables. pip install openpyxl"
            ) from None
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        it = ws.iter_rows(values_only=True)
        header = ["" if h is None else str(h).strip() for h in next(it)]
        rows = [dict(zip(header, r)) for r in it]
        wb.close()
        return rows
    if suffix == ".xls":
        raise SystemExit(f"{path.name} is legacy .xls; re-save it as .xlsx")
    delim = "\t" if suffix in {".tsv", ".tab"} else ","
    with path.open(newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        if suffix not in {".tsv", ".tab"}:
            try:
                delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                pass
        return list(csv.DictReader(fh, delimiter=delim))


def _match_columns(headings: list[str]) -> dict[str, str]:
    """Map our required fields onto the table's actual column names."""
    norm = {_norm(h): h for h in headings if h and str(h).strip()}
    resolved: dict[str, str] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in norm:
                resolved[field] = norm[alias]
                break
        else:
            # Fall back to substring containment, but only if it is unambiguous.
            hits = [orig for n, orig in norm.items()
                    if any(a in n for a in aliases)]
            if len(hits) == 1:
                resolved[field] = hits[0]
    return resolved


def index_videos(root: Path) -> dict[str, str]:
    """Map clip stem -> a video reference, from a directory OR a .zip.

    A zip is indexed from its central directory and the clips are referenced as
    `<archive>::<member>`, which extract_pose.py reads directly. The release
    ships 17 GB of Signer-Only Communication clips whose deflate ratio is 0.997
    -- video is already compressed, so the archive is doing no real work and
    unpacking it would cost the disk twice over for nothing.
    """
    index: dict[str, str] = {}
    collisions = 0

    if root.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(root) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                name = Path(member)
                if name.suffix.lower() in VIDEO_EXTS:
                    if name.stem in index:
                        collisions += 1
                    index[name.stem] = f"{root}::{member}"
    else:
        for p in root.rglob("*"):
            if p.suffix.lower() in VIDEO_EXTS:
                if p.stem in index:
                    collisions += 1
                index[p.stem] = str(p)

    if collisions:
        print(f"  warning: {collisions} clip names appear more than once in "
              f"{root}; the last one wins.", file=sys.stderr)
    return index


def _lookup(videos: dict[str, str], stem: str) -> tuple[str | None, str]:
    """Resolve a table id to a video reference. Returns (ref, which rule)."""
    if stem in videos:
        return videos[stem], "exact"
    with_suffix = stem + SIGNER_SUFFIX
    if with_suffix in videos:
        return videos[with_suffix], "signer_suffix"
    return None, "missing"


def build_records(split_table: Path, video_root: Path, subset: str,
                  default_split: str | None = None) -> list[dict]:
    rows = _read_table(split_table)
    if not rows:
        raise SystemExit(f"{split_table} is empty")

    cols = _match_columns(list(rows[0].keys()))
    print(f"  columns matched: {cols}")
    for required in ("id", "text"):
        if required not in cols:
            raise SystemExit(
                f"could not find a {required!r} column in {split_table.name}.\n"
                f"  headings present: {list(rows[0].keys())}\n"
                f"  add the right heading to COLUMN_ALIASES[{required!r}] in "
                "auslan_daily.py rather than renaming the release's file."
            )
    if "split" not in cols and not default_split:
        raise SystemExit(
            f"no split column in {split_table.name} and no --default-split "
            "given. Refusing to invent a split: an accidental train/test "
            "overlap is not visible in any metric."
        )

    videos = index_videos(video_root)
    print(f"  {len(videos)} video files indexed in {video_root}")

    records, missing, bad_split = [], [], Counter()
    matched_by = Counter()
    for row in rows:
        clip = str(row[cols["id"]]).strip()
        if not clip:
            continue
        stem = Path(clip).stem
        text = str(row[cols["text"]]).strip()
        if not text:
            continue

        raw_split = (str(row[cols["split"]]).strip().lower()
                     if "split" in cols else default_split)
        split = SPLIT_ALIASES.get(_norm(raw_split or ""), None)
        if split is None:
            bad_split[raw_split] += 1
            continue

        video, rule = _lookup(videos, stem)
        matched_by[rule] += 1
        if video is None:
            missing.append(stem)
            continue

        records.append({
            "id": stem,
            "video": video,          # a path, or "<archive>.zip::<member>"
            "text": text,
            "split": split,
            "subset": subset,
        })

    if bad_split:
        print(f"  unrecognised split labels skipped: {dict(bad_split)}",
              file=sys.stderr)
    if missing:
        print(f"  {len(missing)} rows had no matching video file "
              f"(e.g. {missing[:5]})", file=sys.stderr)
    counts = Counter(r["split"] for r in records)
    print(f"  matched by: {dict(matched_by)}")
    print(f"  {subset}: {len(records)} records {dict(counts)}")
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split-table", type=Path, required=True, action="append",
                    help="the release's split table (CSV/TSV/JSON); repeatable")
    ap.add_argument("--video-root", type=Path, required=True, action="append",
                    help="the Signer.zip archive (read in place, no unpacking) "
                         "or a folder of Signer-Only clips; paired with each "
                         "--split-table in order")
    ap.add_argument("--subset", required=True, action="append",
                    choices=["communication", "news"],
                    help="which sub-dataset each table describes, in order")
    ap.add_argument("--default-split", default=None,
                    help="only if the table has no split column")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if not (len(args.split_table) == len(args.video_root) == len(args.subset)):
        ap.error("--split-table, --video-root and --subset must be repeated "
                 "the same number of times, in matching order")

    records: list[dict] = []
    for table, root, subset in zip(args.split_table, args.video_root, args.subset):
        print(f"{subset}: {table}")
        records += build_records(table, root, subset, args.default_split)

    if not records:
        raise SystemExit("no records produced")

    # A clip appearing in two splits is the one error that no metric reveals.
    by_id: dict[str, set] = {}
    for r in records:
        by_id.setdefault(r["id"], set()).add(r["split"])
    leaked = {k: v for k, v in by_id.items() if len(v) > 1}
    if leaked:
        raise SystemExit(
            f"{len(leaked)} clips appear in more than one split, e.g. "
            f"{list(leaked.items())[:3]}. Refusing to write a manifest with "
            "train/test leakage."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=1, ensure_ascii=False))
    print(f"\n{len(records)} records -> {args.out}")
    print("Now: python manifest.py --out data/manifest.jsonl "
          f"--mmwl-root ../MM-WLAuslan --ad-annotations {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
