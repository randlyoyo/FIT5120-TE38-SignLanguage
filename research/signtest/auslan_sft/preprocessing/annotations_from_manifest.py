#!/usr/bin/env python3
"""Auslan-Daily adapter: unisign manifest.jsonl -> annotations.csv.

The existing sign->text project (../unisign) already extracted rtmlib poses for
all 25,109 Auslan-Daily clips, named <uid>.npz. This writes an annotations CSV
whose `uid` column matches those files, so with

    data.id_column: uid
    data.raw_pose_dir: <directory holding the restored .npz files>

extraction can be skipped. The official split is kept in `split`; use
`split.method: column` if you intend to back-translate with an SLR model that
was trained on Auslan-Daily's official train split (the ../unisign runs were).

Excluded clips (unisign excluded.txt: mirrored / wrong-person tracking) can be
dropped with --exclude.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="auslandaily")
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="e.g. communication news (default: all)")
    ap.add_argument("--exclude", default=None, help="text file of uids to drop")
    args = ap.parse_args(argv)

    excluded = set()
    if args.exclude:
        excluded = {l.strip() for l in Path(args.exclude).read_text().splitlines() if l.strip()}
    n_in = n_out = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as fh, open(args.out, "w", newline="", encoding="utf-8") as oh:
        w = csv.DictWriter(oh, fieldnames=["uid", "video_path", "text", "split", "subset"])
        w.writeheader()
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("dataset") != args.dataset:
                continue
            n_in += 1
            if args.subsets and r.get("subset") not in args.subsets:
                continue
            if r["uid"] in excluded or not (r.get("text") or "").strip():
                continue
            w.writerow({"uid": r["uid"], "video_path": r["video"], "text": r["text"].strip(),
                        "split": r.get("split", ""), "subset": r.get("subset", "")})
            n_out += 1
    print(f"{n_out}/{n_in} {args.dataset} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
