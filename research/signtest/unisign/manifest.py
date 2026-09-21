#!/usr/bin/env python3
"""Build one manifest format for both datasets.

Everything downstream reads manifests, never directory layouts. The two corpora
are shaped nothing alike -- MM-WLAuslan is 45k short clips inside zip archives
keyed by a numeric id, Auslan-Daily is continuous video with sentence
alignments -- and the only way the rest of the pipeline can honestly share one
preprocessing path (section 3.3 of the plan) is if that difference is absorbed
here and nowhere else.

Manifest is JSONL, one clip per line:

    uid      unique across both datasets; used as the .npz filename
    dataset  "mmwlauslan" | "auslandaily"
    subset   "studio" | "communication" | "news" | ...
    split    "train" | "val" | "test"
    video    path to an .mp4, or "<archive.zip>::<member>" for a zipped clip
    text     the target string the SLT head is trained against
    gloss    isolated-sign gloss, or null for continuous data
    camera   camera tag if the corpus records one, else null

`text` is what the model is trained to produce. For MM-WLAuslan it is the gloss
itself: the plan trains the isolated data in SLT form rather than as a
classification head, so the target has to be a string.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

# MM-WLAuslan uses two filename conventions and the label keys follow them:
#   studio splits  <id>_<camera>_rgb.mp4   e.g. 56893_kf_rgb.mp4, key "56893"
#   Test_MTV       <camera>_<id>.mp4       e.g. web1_832.mp4,     key "web1_832"
# Both are matched so that MTV is dropped by the camera filter -- a deliberate
# exclusion -- rather than reported as missing, which would look like a broken
# download.
_MMWL_CLIP = re.compile(r"^(?P<cid>\d+)_(?P<cam>[A-Za-z0-9]+)_rgb\.mp4$")
_MMWL_MTV_CLIP = re.compile(r"^(?P<cam>[A-Za-z]*\d+)_(?P<num>\d+)\.mp4$")


def _label_key(member: str) -> tuple[str, str] | None:
    """Archive member -> (label key, camera tag), or None if not a clip."""
    name = Path(member).name
    m = _MMWL_CLIP.match(name)
    if m:
        return m["cid"], m["cam"]
    m = _MMWL_MTV_CLIP.match(name)
    if m:
        return name[: -len(".mp4")], m["cam"]
    return None

# The studio frontal camera. Train/Valid/Test_STU/ITW/TED/SYN contain only this
# one; Test_MTV is the multi-device set (phone*/web*) and is excluded by
# default, per the plan's decision to train and deploy frontal-only.
MMWL_FRONTAL_CAMERA = "kf"

MMWL_SPLIT_MAP = {
    "Train": ("train", "studio"),
    "Valid": ("val", "studio"),
    "Test_STU": ("test", "studio"),
    "Test_ITW": ("test", "in_the_wild"),
    "Test_SYN": ("test", "synthetic_bg"),
    "Test_TED": ("test", "tv_studio"),
    "Test_MTV": ("test", "multi_view"),
}


# MM-WLAuslan ships in two shapes and both are in the wild: the flat one this
# project first met locally, and the full Drive release. They differ in the
# label directory name, in hyphen-vs-underscore split names, and in whether the
# archives sit under a per-camera folder. Rather than make the caller normalise
# a 200 GB tree, both are recognised here.
#
# The Drive release also settles the camera question outright: its folders are
# named Kinect_F, Kinect_L, Kinect_R and RealSense_F, so the `kf` tag in the
# clip filenames is Kinect Front. Frontal-only means Kinect_F.
_LABEL_DIRS = ("labels", "Annotation/Labels & Split")
_FRONTAL_CAMERA_DIRS = ("Kinect_F",)


def _find_label_dir(root: Path) -> Path:
    for rel in _LABEL_DIRS:
        d = root / rel
        if d.is_dir():
            return d
    raise FileNotFoundError(
        f"no label directory under {root}; looked for "
        + ", ".join(repr(r) for r in _LABEL_DIRS)
    )


def _split_dir_candidates(root: Path, split_name: str) -> list[Path]:
    names = {split_name, split_name.replace("_", "-")}
    out = []
    for n in names:
        base = root / n
        out.append(base)
        out.extend(base / cam for cam in _FRONTAL_CAMERA_DIRS)
    return out


def _find_archive(root: Path, split_name: str) -> Path:
    """The frontal RGB archive for one split, in either release layout.

    depth.zip is excluded rather than reported as ambiguity: this project is
    pose-only from a single frontal view, and the depth stream is never used.
    """
    seen = []
    for d in _split_dir_candidates(root, split_name):
        if not d.is_dir():
            continue
        seen.append(d)
        zips = [z for z in sorted(d.glob("*.zip")) if "depth" not in z.name.lower()]
        if len(zips) == 1:
            return zips[0]
        if len(zips) > 1:
            rgb = [z for z in zips if "rgb" in z.name.lower()]
            if len(rgb) == 1:
                return rgb[0]
            raise RuntimeError(
                f"{d}: cannot choose between " + ", ".join(z.name for z in zips)
            )
    raise FileNotFoundError(
        f"no RGB archive for split {split_name!r}; looked in "
        + (", ".join(str(d) for d in seen) if seen else
           ", ".join(str(d) for d in _split_dir_candidates(root, split_name)))
    )


def build_mmwlauslan(
    root: Path, camera: str | None = MMWL_FRONTAL_CAMERA, splits: list[str] | None = None
) -> list[dict]:
    """Manifest rows for MM-WLAuslan, read from labels/*.json + the archives.

    The label json is the authority on which clips exist; the archive is
    consulted to resolve each id to a real member. An id present in the labels
    but missing from the archive is reported rather than silently dropped --
    a partial download is otherwise indistinguishable from a small dataset.
    """
    label_dir = _find_label_dir(root)

    wanted = splits or list(MMWL_SPLIT_MAP)
    rows: list[dict] = []
    for split_name in wanted:
        if split_name not in MMWL_SPLIT_MAP:
            raise KeyError(f"unknown MM-WLAuslan split {split_name!r}")
        label_file = label_dir / f"{split_name}.json"
        if not label_file.exists():
            print(f"  skip {split_name}: no label file", file=sys.stderr)
            continue
        try:
            archive = _find_archive(root, split_name)
        except FileNotFoundError:
            print(f"  skip {split_name}: not downloaded", file=sys.stderr)
            continue

        split, subset = MMWL_SPLIT_MAP[split_name]
        labels: dict[str, str] = json.loads(label_file.read_text())

        # id -> [(member, camera)], from the archive's central directory.
        by_id: dict[str, list[tuple[str, str]]] = {}
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                parsed = _label_key(member)
                if parsed:
                    key, cam = parsed
                    by_id.setdefault(key, []).append((member, cam))

        missing = 0
        filtered = 0
        for clip_id, gloss in labels.items():
            entries = by_id.get(clip_id)
            if not entries:
                missing += 1
                continue
            for member, cam in entries:
                if camera is not None and cam != camera:
                    filtered += 1
                    continue
                rows.append(
                    {
                        "uid": f"mmwl-{split_name}-{clip_id}-{cam}",
                        "dataset": "mmwlauslan",
                        "subset": subset,
                        "split": split,
                        "video": f"{archive}::{member}",
                        "text": gloss,
                        "gloss": gloss,
                        "camera": cam,
                    }
                )
        note = f"  {split_name}: {len(labels)} labelled"
        if missing:
            note += f", {missing} MISSING FROM ARCHIVE"
        if filtered:
            note += f", {filtered} dropped by camera filter"
        print(note, file=sys.stderr)
    return rows


# --- Auslan-Daily -----------------------------------------------------------
#
# Reads the records JSON that auslan_daily.py produces from the release:
#
#   [{"id": "...", "video": "clips/xyz.mp4", "text": "the english sentence",
#     "split": "train"|"val"|"test", "subset": "communication"|"news"}, ...]
#
# The release ships two sub-datasets, several video variants and a split table
# whose column headings are not published, so that normalisation lives in
# auslan_daily.py and this loader stays a plain reader of one known shape.

_AD_SUBSETS = {"communication", "news"}
_AD_SPLITS = {"train", "val", "test"}


def build_auslan_daily(annotations: Path, video_root: Path | None = None) -> list[dict]:
    """Read the records auslan_daily.py wrote.

    `video` is already resolved there -- an absolute path, or an
    "<archive>.zip::<member>" reference that extract_pose.py opens without
    unpacking. `video_root` is only used for legacy records holding a relative
    path, and is otherwise ignored: re-joining a root onto an already-resolved
    reference is how a zip ref turns into a nonexistent file.
    """
    records = json.loads(Path(annotations).read_text())
    if not isinstance(records, list):
        raise TypeError("annotations must be a JSON list of records")

    rows: list[dict] = []
    for i, rec in enumerate(records):
        for field in ("id", "video", "text", "split", "subset"):
            if field not in rec:
                raise KeyError(f"record {i}: missing {field!r}")
        subset = str(rec["subset"]).lower()
        split = str(rec["split"]).lower()
        if subset not in _AD_SUBSETS:
            raise ValueError(f"record {i}: subset {subset!r} not in {_AD_SUBSETS}")
        if split not in _AD_SPLITS:
            raise ValueError(f"record {i}: split {split!r} not in {_AD_SPLITS}")
        text = str(rec["text"]).strip()
        if not text:
            raise ValueError(f"record {i}: empty text")
        video = str(rec["video"])
        if "::" not in video and not Path(video).is_absolute():
            if video_root is None:
                raise ValueError(
                    f"record {i}: relative video path {video!r} and no "
                    "--ad-video-root given"
                )
            video = str(video_root / video)

        rows.append(
            {
                "uid": f"ad-{subset}-{rec['id']}",
                "dataset": "auslandaily",
                "subset": subset,
                "split": split,
                "video": video,
                "text": text,
                "gloss": None,
                "camera": None,
            }
        )
    return rows


def write_manifest(rows: list[dict], out: Path) -> None:
    seen: set[str] = set()
    for r in rows:
        if r["uid"] in seen:
            raise ValueError(f"duplicate uid {r['uid']!r}")
        seen.add(r["uid"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarise(rows: list[dict]) -> str:
    from collections import Counter

    by = Counter((r["dataset"], r["subset"], r["split"]) for r in rows)
    vocab = Counter(r["gloss"] for r in rows if r["gloss"])
    lines = [f"{len(rows)} clips"]
    for (ds, sub, sp), n in sorted(by.items()):
        lines.append(f"  {ds:<12} {sub:<14} {sp:<6} {n:>7}")
    if vocab:
        lines.append(f"  distinct glosses: {len(vocab)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="manifest .jsonl to write")
    ap.add_argument("--mmwl-root", type=Path, help="MM-WLAuslan directory")
    ap.add_argument(
        "--mmwl-splits", nargs="*", default=None,
        help=f"subset of {list(MMWL_SPLIT_MAP)} (default: all present)",
    )
    ap.add_argument(
        "--mmwl-camera", default=MMWL_FRONTAL_CAMERA,
        help="keep only this camera tag; 'all' to keep every view",
    )
    ap.add_argument("--ad-annotations", type=Path, help="Auslan-Daily records json")
    ap.add_argument("--ad-video-root", type=Path, default=None,
                    help="only needed if the records hold relative paths; "
                         "auslan_daily.py already resolves them")
    args = ap.parse_args(argv)

    rows: list[dict] = []
    if args.mmwl_root:
        cam = None if args.mmwl_camera == "all" else args.mmwl_camera
        rows += build_mmwlauslan(args.mmwl_root, cam, args.mmwl_splits)
    if args.ad_annotations:
        rows += build_auslan_daily(args.ad_annotations, args.ad_video_root)
    if not rows:
        ap.error("nothing to do: pass --mmwl-root and/or --ad-annotations")

    write_manifest(rows, args.out)
    print(summarise(rows))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
