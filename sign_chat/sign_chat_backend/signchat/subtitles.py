"""Subtitles for the avatar's reply: the reply text cut into short cues, timed to the signing.

The signing of one reply is a single clip whose length the model sets from the word count
(frames = a + b x words), so there is no word-to-frame alignment to read off. The cues share the
clip's duration in proportion to their word counts, which follows the same rule, and together cover
the whole clip with no gaps. `offset` moves them to where the signing starts in the reply (after the
avatar has eased out of its rest pose, stage.py).

    cues("Hello! I am good, thank you. And you?", frames=76, fps=25)
    -> [{"start": 0.0, "end": 0.38, "text": "Hello!"}, {"start": 0.38, ...}, ...]

Each cue holds one sentence, or part of one: sentences longer than `max_words` are cut at commas,
then into even chunks of at most `max_words` words.
"""

from __future__ import annotations

import math
import os
import re

MAX_WORDS = 7


def split_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    parts = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sentence.split()
        if not words:
            continue
        if len(words) <= max_words:
            parts.append(sentence)
            continue
        for clause in re.split(r"(?<=[,;:])\s+", sentence):
            w = clause.split()
            n = math.ceil(len(w) / max_words)              # even chunks: 9 words -> 5 + 4, not 7 + 2
            size = math.ceil(len(w) / n) if n else 0
            parts += [" ".join(w[i:i + size]) for i in range(0, len(w), size)] if w else []
    return parts


def cues(text: str, frames: int, fps: float, max_words: int = MAX_WORDS, offset: float = 0.0) -> list[dict]:
    parts = split_text(text, max_words)
    if not parts or frames <= 0:
        return []
    weights = [len(p.split()) for p in parts]
    total, duration = sum(weights), frames / fps
    out, done = [], 0
    for p, w in zip(parts, weights):
        start = duration * done / total
        done += w
        out.append({"start": round(offset + start, 3), "end": round(offset + duration * done / total, 3), "text": p})
    return out


def per_frame(cue_list: list[dict], frames: int, fps: float) -> list[str]:
    """The cue shown on each frame of the video ('' where none is)."""
    out = [""] * frames
    for c in cue_list:
        for t in range(int(round(c["start"] * fps)), min(frames, int(round(c["end"] * fps)))):
            out[t] = c["text"]
    return out


def _stamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def to_vtt(cue_list: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for i, c in enumerate(cue_list, 1):
        lines += [str(i), f"{_stamp(c['start'])} --> {_stamp(c['end'])}", c["text"], ""]
    return "\n".join(lines)


def write(out_dir: str, name: str, text: str, frames: int, fps: float, offset: float = 0.0) -> tuple[list[dict], str]:
    """Cues for `text` over `frames` frames of signing that start `offset` seconds into the reply,
    also written to <out_dir>/<name>.vtt. Returns (cues, the .vtt file name)."""
    cue_list = cues(text, frames, fps, offset=offset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.vtt"), "w", encoding="utf-8") as fh:
        fh.write(to_vtt(cue_list))
    return cue_list, f"{name}.vtt"
