#!/usr/bin/env python3
"""Talk to the avatar from the terminal, without the HTTP server (loads the same pipeline).

    python scripts/demo_cli.py --config config.yaml                 # type messages; ':sign <video>' sends a video
    python scripts/demo_cli.py --mock                               # no models, checks the plumbing
    python scripts/demo_cli.py --config config.yaml --once "Hello, how are you?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from signchat.config import load_config  # noqa: E402
from signchat.pipeline import ChatPipeline  # noqa: E402


def show(r: dict) -> None:
    i, rep = r["input"], r["reply"]
    print(f"  you ({i['mode']}): {i['text']}")
    print(f"  avatar : {rep['text']}")
    s = rep["sign"]
    print(f"  signing: {s['frames']} frames @ {s['fps']} fps | pose {s.get('pose_file')} | video {s.get('video_file')}")
    if s.get("retrieved"):
        print(f"           hands from training sentence: {s['retrieved']}{'  (exact match)' if s['seen'] else ''}")
    print(f"  timings: {json.dumps(r['timings'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dialogue", help="override dialogue.backend (hf / openai / anthropic / echo)")
    ap.add_argument("--once", help="send one message and exit")
    args = ap.parse_args()
    over = {"mock": True} if args.mock else {}
    if args.dialogue:
        over["dialogue"] = {"backend": args.dialogue}
    pipe = ChatPipeline(load_config(args.config, over))
    print(json.dumps(pipe.status(), indent=2))
    sid = "cli"
    if args.once:
        show(pipe.turn(sid, text=args.once))
        return 0
    print("type a message, ':sign <video path> [mirrored]', ':reset', or ':quit'")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in (":quit", ":q"):
            break
        try:
            if line == ":reset":
                pipe.sessions.clear(sid)
                print("  (conversation cleared)")
            elif line.startswith(":sign "):
                parts = line.split()
                show(pipe.turn(sid, video_path=parts[1], mirrored="mirrored" in parts[2:]))
            else:
                show(pipe.turn(sid, text=line))
        except (ValueError, RuntimeError) as e:
            print(f"  error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
