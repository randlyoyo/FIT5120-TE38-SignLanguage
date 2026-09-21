#!/usr/bin/env python3
"""Hand-built lattices for exercising the translation prompt on its own.

These exist so the prompt can be judged before it is wired to a real
recogniser. When the whole pipeline is live and a translation comes out wrong,
there is no way to tell whether the recogniser proposed bad candidates or the
prompt mishandled good ones. Here the candidates are known, so anything wrong
in the output is the prompt's doing.

Each fixture isolates one thing the prompt claims to handle. Confidences are
written to look like recogniser output -- a clear winner where the sign is
distinct, a close pair where it is not -- rather than being uniformly high.

    python fixtures.py                    # list them
    python fixtures.py confusable         # dump one as JSON
    python fixtures.py --write out/       # write all four as .json files
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Every candidate list is longer than the winner, because a recogniser always
# returns a ranked list and a prompt that only ever sees one plausible option
# is not being tested on selection at all.

fixture_minimal: dict[str, Any] = {
    "segments": [
        {
            "id": 0,
            "t": [0.00, 0.62],
            "candidates": [
                {"gloss": "YESTERDAY", "conf": 0.94},
                {"gloss": "LAST-WEEK", "conf": 0.21},
                {"gloss": "BEFORE", "conf": 0.14},
            ],
            "note": None,
        },
        {
            "id": 1,
            "t": [0.62, 1.35],
            "candidates": [
                {"gloss": "MOTHER", "conf": 0.91},
                {"gloss": "FATHER", "conf": 0.18},
                {"gloss": "WOMAN", "conf": 0.11},
            ],
            "note": None,
        },
        {
            "id": 2,
            "t": [1.35, 2.20],
            "candidates": [
                {"gloss": "COOK", "conf": 0.89},
                {"gloss": "EAT", "conf": 0.23},
                {"gloss": "KITCHEN", "conf": 0.12},
            ],
            "note": None,
        },
    ]
}
"""Three clean segments. No selection difficulty, no dropped segments.

The readings should still diverge -- YESTERDAY MOTHER COOK is a statement or a
yes/no question depending on brow position, which the input does not carry --
so this fixture checks that the prompt reports that ambiguity even when nothing
about the glosses themselves is in doubt.
"""


fixture_confusable: dict[str, Any] = {
    "segments": [
        {
            "id": 0,
            "t": [0.00, 0.71],
            "candidates": [
                {"gloss": "LAST-WEEK", "conf": 0.90},
                {"gloss": "YESTERDAY", "conf": 0.26},
                {"gloss": "LAST-YEAR", "conf": 0.17},
            ],
            "note": None,
        },
        {
            "id": 1,
            "t": [0.71, 1.18],
            "candidates": [
                {"gloss": "PRO1", "conf": 0.88},
                {"gloss": "PRO3", "conf": 0.24},
                {"gloss": "POINT", "conf": 0.15},
            ],
            "note": None,
        },
        {
            "id": 2,
            "t": [1.18, 1.94],
            "candidates": [
                {"gloss": "SHOP", "conf": 0.86},
                {"gloss": "MARKET", "conf": 0.31},
                {"gloss": "BUILDING", "conf": 0.13},
            ],
            "note": None,
        },
        {
            "id": 3,
            "t": [1.94, 2.55],
            "candidates": [
                {"gloss": "GO-TO", "conf": 0.83},
                {"gloss": "ARRIVE", "conf": 0.29},
                {"gloss": "WALK", "conf": 0.19},
            ],
            "note": None,
        },
        {
            "id": 4,
            "t": [2.55, 3.40],
            "candidates": [
                {"gloss": "BUY", "conf": 0.68},
                {"gloss": "SELL", "conf": 0.65},
                {"gloss": "PAY", "conf": 0.44},
                {"gloss": "GIVE", "conf": 0.22},
            ],
            "note": None,
        },
    ]
}
"""BUY and SELL are near-identical in the hands and the recogniser cannot
separate them: 0.68 against 0.65 is noise, not a decision.

The surrounding context can. A first-person subject going to a shop buys there;
selling would be odd. This checks that the prompt uses the sentence to pick
between candidates instead of taking whichever number is larger -- and that it
says so in the segment's reason.
"""


fixture_noisy: dict[str, Any] = {
    "segments": [
        {
            "id": 0,
            "t": [0.00, 0.68],
            "candidates": [
                {"gloss": "TOMORROW", "conf": 0.92},
                {"gloss": "FUTURE", "conf": 0.28},
                {"gloss": "NEXT-WEEK", "conf": 0.19},
            ],
            "note": None,
        },
        {
            "id": 1,
            "t": [0.68, 0.83],
            "candidates": [
                {"gloss": "POINT", "conf": 0.41},
                {"gloss": "PRO3", "conf": 0.37},
                {"gloss": "THERE", "conf": 0.33},
            ],
            "note": None,
        },
        {
            "id": 2,
            "t": [0.83, 1.61],
            "candidates": [
                {"gloss": "MOTHER", "conf": 0.87},
                {"gloss": "GRANDMOTHER", "conf": 0.25},
                {"gloss": "WOMAN", "conf": 0.16},
            ],
            "note": None,
        },
        {
            "id": 3,
            "t": [1.61, 1.98],
            "candidates": [
                {"gloss": "MOVE", "conf": 0.24},
                {"gloss": "HAVE", "conf": 0.21},
                {"gloss": "PUT", "conf": 0.18},
            ],
            "note": "low_confidence_all",
        },
        {
            "id": 4,
            "t": [1.98, 2.79],
            "candidates": [
                {"gloss": "HOSPITAL", "conf": 0.81},
                {"gloss": "DOCTOR", "conf": 0.35},
                {"gloss": "SICK", "conf": 0.20},
            ],
            "note": None,
        },
        {
            "id": 5,
            "t": [2.79, 3.46],
            "candidates": [
                {"gloss": "GO-TO", "conf": 0.85},
                {"gloss": "TRAVEL", "conf": 0.27},
                {"gloss": "VISIT", "conf": 0.24},
            ],
            "note": None,
        },
    ]
}
"""Two segments that should not survive, for two different reasons.

Segment 1 lasts 0.15s -- too short to be a sign, and its candidates are the
pointing-like shapes a hand passes through on the way somewhere. Segment 3 is
flagged low_confidence_all: nothing scored well because nothing was there.

Dropping both is the intended behaviour, and the prompt permits it. Keeping
either would put a spurious word in the sentence, so this fixture is really
asking whether "you MAY drop a segment" is acted on or merely acknowledged.
"""


fixture_oov: dict[str, Any] = {
    "segments": [
        {
            "id": 0,
            "t": [0.00, 0.59],
            "candidates": [
                {"gloss": "YESTERDAY", "conf": 0.93},
                {"gloss": "LAST-NIGHT", "conf": 0.30},
                {"gloss": "BEFORE", "conf": 0.15},
            ],
            "note": None,
        },
        {
            "id": 1,
            "t": [0.59, 1.02],
            "candidates": [
                {"gloss": "PRO1", "conf": 0.90},
                {"gloss": "PRO3", "conf": 0.22},
                {"gloss": "POINT", "conf": 0.14},
            ],
            "note": None,
        },
        {
            "id": 2,
            "t": [1.02, 1.88],
            "candidates": [],
            "note": "out_of_vocabulary",
        },
        {
            "id": 3,
            "t": [1.88, 2.61],
            "candidates": [
                {"gloss": "EAT", "conf": 0.88},
                {"gloss": "FOOD", "conf": 0.34},
                {"gloss": "HUNGRY", "conf": 0.18},
            ],
            "note": None,
        },
    ]
}
"""A real sign the vocabulary cannot name, sitting where the object belongs.

The temptation is to fill the hole -- YESTERDAY PRO1 ... EAT invites a guess at
a food, and any guess would read fluently. The prompt forbids that: the slot
stays [UNKNOWN], and any inference about what it plausibly was belongs in the
notes, not in the sentence. This is the fixture most likely to catch a prompt
that has drifted towards fluency at the cost of honesty.
"""


FIXTURES: dict[str, dict[str, Any]] = {
    "minimal": fixture_minimal,
    "confusable": fixture_confusable,
    "noisy": fixture_noisy,
    "oov": fixture_oov,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", nargs="?", choices=sorted(FIXTURES))
    ap.add_argument("--write", metavar="DIR", help="write all fixtures as JSON")
    args = ap.parse_args()

    if args.write:
        from pathlib import Path
        d = Path(args.write)
        d.mkdir(parents=True, exist_ok=True)
        for name, lat in FIXTURES.items():
            p = d / f"{name}.json"
            p.write_text(json.dumps(lat, indent=2) + "\n", encoding="utf-8")
            print(p)
        return 0

    if args.name:
        print(json.dumps(FIXTURES[args.name], indent=2))
        return 0

    for name, lat in FIXTURES.items():
        segs = lat["segments"]
        flags = {s["note"] for s in segs if s["note"]}
        print(f"{name:12s} {len(segs)} segments, "
              f"{segs[-1]['t'][1]:.2f}s"
              + (f", flags: {sorted(flags)}" if flags else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
