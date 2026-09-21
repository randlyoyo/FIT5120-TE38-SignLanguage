#!/usr/bin/env python3
"""Translate a sign-recognition lattice into English sentences with an LLM.

The recogniser emits a lattice -- segments in time order, each with ranked
candidate glosses -- and this module asks a model to pick a path through it and
render that path as English.

The validation in `validate` is the point of the module. An LLM asked to
translate glosses will readily produce a fluent sentence containing a gloss the
recogniser never proposed, and the result reads perfectly. Every check here
exists to catch a specific way the output can be plausible and wrong.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-sonnet-5"
UNKNOWN = "[UNKNOWN]"

VALID_ACTIONS = {"selected", "dropped", "merged", "unknown"}
VALID_READINGS = {
    "statement", "yes-no question", "wh-question", "negation", "conditional",
}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_NOTES = {"low_confidence_all", "out_of_vocabulary", None}


class ParseError(Exception):
    """The model's response was not the JSON object this module expects."""


class ValidationError(Exception):
    """The model's response broke a constraint and repair did not fix it.

    Carries `violations` and the `raw` response: a rejected response is the
    only evidence of what went wrong, and discarding it makes the failure
    impossible to diagnose.
    """

    def __init__(self, violations: list[str], raw: str) -> None:
        super().__init__("; ".join(violations))
        self.violations = violations
        self.raw = raw


TRANSLATION_PROMPT = """You are an expert Auslan (Australian Sign Language) linguist and translator.

You will receive the output of a sign language recognition system: a continuous
signed utterance that has been segmented into units, where each unit has several
candidate glosses ranked by recognition confidence. Your task is to choose the
most plausible gloss sequence and translate it into natural English.

## Input format

{
  "segments": [
    {
      "id": <int>,
      "t": [<start_sec>, <end_sec>],
      "candidates": [{"gloss": "<GLOSS>", "conf": <0.0-1.0>}, ...],
      "note": "<optional flag>"
    }
  ]
}

Notes you may see:
- "low_confidence_all": every candidate scored poorly. This segment is likely a
  transitional movement between signs, not a sign itself.
- "out_of_vocabulary": nothing in the reference vocabulary matched. A real sign
  was produced here, but the system cannot name it.

## Hard constraints

1. For each segment you keep, you MUST select a gloss from that segment's own
   candidate list. Never invent a gloss, never borrow one from another segment.
2. You MAY drop a segment entirely if it is a transitional movement. Very short
   segments (under ~0.2s) and segments flagged "low_confidence_all" are the
   usual cases.
3. You MAY merge adjacent segments if over-segmentation appears to have split a
   single sign in two.
4. Never silently fill an "out_of_vocabulary" gap with a guessed gloss. Keep it
   as [UNKNOWN] in the gloss sequence. You may infer from context what kind of
   word it plausibly was, but say so explicitly in your reasoning.

## Auslan grammar to apply

- Time markers (YESTERDAY, TOMORROW, LAST-WEEK) are typically utterance-initial.
- Location markers usually precede the main predicate.
- Topic-comment structure is common: the topic is fronted, then commented on.
- Wh-words (WHAT, WHERE, WHO, WHY) typically appear utterance-finally.
- Articles, auxiliaries, copulas, and tense inflection are generally NOT signed.
  You will need to add them in English.
- Negation may follow the verb.

## Critical limitation you must account for

The input contains ONLY manual features (hand and body movement). Non-manual
features - eyebrow position, head tilt, body lean, mouth patterns - were NOT
captured. In Auslan these carry:
  - yes/no questions (raised brows)
  - wh-questions (furrowed brows)
  - negation (headshake)
  - conditionals (brow raise + head tilt)
  - topic marking

This means an identical gloss sequence may correspond to a statement, a
question, or a negation, and you cannot tell which from the input alone. When a
sequence is ambiguous in this way, your alternative readings MUST cover the
different possibilities rather than committing to one.

Spatial information is also absent, so directional verbs (GIVE, SHOW, TELL) do
not indicate who acted on whom. Flag this when it affects the reading.

## Output format

Return JSON only. No markdown fences, no commentary outside the JSON.

{
  "gloss_sequence": ["<GLOSS>", ...],
  "segment_decisions": [
    {
      "id": <int>,
      "action": "selected" | "dropped" | "merged" | "unknown",
      "gloss": "<chosen gloss or null>",
      "reason": "<brief: why this candidate over the others, or why dropped>"
    }
  ],
  "translations": [
    {
      "text": "<English sentence>",
      "reading": "<statement | yes-no question | wh-question | negation | conditional>",
      "confidence": "<high | medium | low>",
      "added_words": ["<words you supplied that were not in the gloss sequence>"],
      "alignment": [{"english": "<word or phrase>", "gloss": "<GLOSS or null>"}]
    }
  ],
  "ambiguity_notes": ["<what could not be determined and why>"]
}

Provide 2-3 translations. Order them most to least likely. If the gloss sequence
is ambiguous between a statement and a question, the alternatives must reflect
that. Set "gloss": null in alignment for words you added."""


# ---------------------------------------------------------------------------
# 1. loading
# ---------------------------------------------------------------------------

def load_lattice(path: str) -> dict[str, Any]:
    """Read and validate a lattice JSON file.

    Every error names the segment and the field, because a lattice is machine
    output and "invalid lattice" tells whoever produced it nothing.
    """
    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "segments" not in data:
        raise ValueError(f"{path}: top level must be an object with 'segments'")
    segments = data["segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{path}: 'segments' must be a non-empty list")

    seen_ids: set[int] = set()
    for pos, seg in enumerate(segments):
        where = f"segment at index {pos}"
        if not isinstance(seg, dict):
            raise ValueError(f"{where}: must be an object")

        sid = seg.get("id")
        if not isinstance(sid, int) or isinstance(sid, bool):
            raise ValueError(f"{where}: 'id' must be an int, got {sid!r}")
        if sid in seen_ids:
            raise ValueError(f"segment {sid}: 'id' is duplicated")
        seen_ids.add(sid)
        where = f"segment {sid}"

        t = seg.get("t")
        if (not isinstance(t, list) or len(t) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           for v in t)):
            raise ValueError(f"{where}: 't' must be two numbers, got {t!r}")
        if not t[0] < t[1]:
            raise ValueError(f"{where}: 't' must increase, got {t!r}")

        note = seg.get("note")
        if note not in VALID_NOTES:
            raise ValueError(
                f"{where}: 'note' must be one of "
                f"{sorted(n for n in VALID_NOTES if n)} or null, got {note!r}"
            )

        cands = seg.get("candidates")
        if not isinstance(cands, list):
            raise ValueError(f"{where}: 'candidates' must be a list")
        if not cands and note != "out_of_vocabulary":
            raise ValueError(
                f"{where}: 'candidates' is empty but 'note' is {note!r}; only "
                "an out_of_vocabulary segment may have no candidates"
            )
        for ci, cand in enumerate(cands):
            cwhere = f"{where}, candidate {ci}"
            if not isinstance(cand, dict):
                raise ValueError(f"{cwhere}: must be an object")
            if not isinstance(cand.get("gloss"), str) or not cand["gloss"]:
                raise ValueError(f"{cwhere}: 'gloss' must be a non-empty string")
            conf = cand.get("conf")
            if (not isinstance(conf, (int, float)) or isinstance(conf, bool)
                    or not 0.0 <= conf <= 1.0):
                raise ValueError(f"{cwhere}: 'conf' must be in [0,1], got {conf!r}")

    return data


# ---------------------------------------------------------------------------
# 2. request
# ---------------------------------------------------------------------------

def build_user_message(lattice: dict[str, Any]) -> str:
    """Serialise the lattice for the model, in time order.

    Confidences are rounded to two places: the extra digits are noise from the
    recogniser and spending tokens on them invites the model to read meaning
    into differences that are not there.
    """
    segments = []
    for seg in sorted(lattice["segments"], key=lambda s: s["t"][0]):
        segments.append({
            "id": seg["id"],
            "t": [round(float(seg["t"][0]), 2), round(float(seg["t"][1]), 2)],
            "candidates": [
                {"gloss": c["gloss"], "conf": round(float(c["conf"]), 2)}
                for c in seg["candidates"]
            ],
            "note": seg.get("note"),
        })
    return json.dumps({"segments": segments}, ensure_ascii=False, indent=2)


def call_llm(
    user_message: str,
    model: str,
    max_retries: int = 2,
    extra_messages: list[dict[str, str]] | None = None,
) -> str:
    """Send one request and return the raw text.

    Retries only transport-level failures. A refusal or a malformed answer is
    the model's considered output and retrying it unchanged just pays twice for
    the same result -- repair, in `translate`, is what handles those.
    """
    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    if extra_messages:
        messages.extend(extra_messages)

    delay = 1.0
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.3,
                system=TRANSLATION_PROMPT,
                messages=messages,
            )
            return "".join(
                block.text for block in resp.content
                if getattr(block, "type", None) == "text"
            )
        except (anthropic.APIConnectionError, anthropic.APITimeoutError,
                anthropic.RateLimitError, anthropic.InternalServerError) as exc:
            last = exc
            if attempt == max_retries:
                break
            print(f"[slt] {type(exc).__name__}, retrying in {delay:.0f}s",
                  file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"LLM request failed after {max_retries + 1} attempts: {last}")


# ---------------------------------------------------------------------------
# 3. parsing
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n?\s*```\s*$", re.S)


def parse_response(raw: str) -> dict[str, Any]:
    """Parse the model's JSON, tolerating a markdown fence around it.

    The prompt forbids fences and the model mostly complies, but "mostly" over
    many calls is a steady trickle of failures for a two-line fix.
    """
    text = raw.strip()
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"response was not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ParseError(f"response was {type(obj).__name__}, expected an object")

    for key in ("gloss_sequence", "segment_decisions", "translations"):
        if key not in obj:
            raise ParseError(f"response is missing '{key}'")
    obj.setdefault("ambiguity_notes", [])
    return obj


# ---------------------------------------------------------------------------
# 4. validation
# ---------------------------------------------------------------------------

def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def validate(result: dict[str, Any], lattice: dict[str, Any]) -> list[str]:
    """Return every constraint the response breaks; empty means it is sound.

    All violations are collected rather than returning on the first, so a
    repair round can fix them together instead of one per round trip.
    """
    v: list[str] = []

    by_id = {s["id"]: s for s in lattice["segments"]}
    cands_by_id = {
        sid: {c["gloss"] for c in s["candidates"]} for sid, s in by_id.items()
    }
    all_cands: set[str] = set().union(*cands_by_id.values()) if cands_by_id else set()

    seq = result.get("gloss_sequence")
    if not isinstance(seq, list):
        return [f"gloss_sequence must be a list, got {type(seq).__name__}"]
    decisions = result.get("segment_decisions")
    if not isinstance(decisions, list):
        return [f"segment_decisions must be a list, got {type(decisions).__name__}"]

    # (a) no invented glosses
    for g in seq:
        if g == UNKNOWN:
            continue
        if g not in all_cands:
            v.append(f"invented gloss: {g}")

    # (b) a selected gloss must come from that segment's own candidates
    for d in decisions:
        if not isinstance(d, dict):
            v.append(f"segment_decisions entry is not an object: {d!r}")
            continue
        sid, action, gloss = d.get("id"), d.get("action"), d.get("gloss")
        if action not in VALID_ACTIONS:
            v.append(f"segment {sid}: unknown action {action!r}")
        if action == "selected":
            if sid not in cands_by_id:
                continue                      # reported by (c)
            if gloss not in cands_by_id[sid]:
                v.append(f"segment {sid} borrowed gloss from elsewhere: {gloss}")

    # (c) decisions must cover exactly the input segments
    decided = {d.get("id") for d in decisions if isinstance(d, dict)}
    missing = sorted(i for i in by_id if i not in decided)
    extra = sorted(i for i in decided if i not in by_id and i is not None)
    if missing:
        v.append(f"segment_decisions missing ids: {missing}")
    if extra:
        v.append(f"segment_decisions has ids not in the lattice: {extra}")

    # (d) kept glosses, in segment time order, must match the sequence.
    # Consecutive duplicates are collapsed: a merge legitimately reports the
    # same gloss on two adjacent segments while contributing one to the
    # sequence.
    kept: list[tuple[float, str]] = []
    for d in decisions:
        if not isinstance(d, dict) or d.get("action") == "dropped":
            continue
        seg = by_id.get(d.get("id"))
        if seg is None:
            continue
        g = UNKNOWN if d.get("action") == "unknown" else d.get("gloss")
        if isinstance(g, str) and g:
            kept.append((seg["t"][0], g))
    kept.sort(key=lambda p: p[0])
    ordered = [g for _, g in kept]
    collapsed = [g for i, g in enumerate(ordered) if i == 0 or g != ordered[i - 1]]
    seq_collapsed = [g for i, g in enumerate(seq) if i == 0 or g != seq[i - 1]]
    if collapsed != seq_collapsed:
        v.append(
            "gloss_sequence order does not match segment time order: "
            f"decisions give {collapsed}, sequence gives {seq_collapsed}"
        )

    translations = result.get("translations")
    if not isinstance(translations, list):
        v.append(f"translations must be a list, got {type(translations).__name__}")
        return v

    # (f) how many translations
    if not 1 <= len(translations) <= 5:
        v.append(f"translations must number 1-5, got {len(translations)}")

    seq_set = set(seq)
    for i, tr in enumerate(translations):
        if not isinstance(tr, dict):
            v.append(f"translation {i} is not an object")
            continue
        text = tr.get("text")
        if not isinstance(text, str) or not text.strip():
            v.append(f"translation {i}: 'text' must be a non-empty string")
            text = ""
        if tr.get("reading") not in VALID_READINGS:
            v.append(f"translation {i}: unknown reading {tr.get('reading')!r}")
        if tr.get("confidence") not in VALID_CONFIDENCE:
            v.append(f"translation {i}: unknown confidence {tr.get('confidence')!r}")

        # (e) an alignment may only point at glosses that are in the sequence
        for a in tr.get("alignment") or []:
            if not isinstance(a, dict):
                v.append(f"translation {i}: alignment entry is not an object")
                continue
            g = a.get("gloss")
            if g is None:
                continue
            if g not in seq_set:
                v.append(
                    f"translation {i}: alignment references {g}, "
                    "which is not in gloss_sequence"
                )

        # (g) an added word must actually be in the sentence
        present = _words(text)
        for w in tr.get("added_words") or []:
            if not isinstance(w, str):
                v.append(f"translation {i}: added_words entry is not a string")
                continue
            if not _words(w) <= present:
                v.append(f"translation {i}: added word {w!r} is not in the text")

    return v


# ---------------------------------------------------------------------------
# 5. orchestration
# ---------------------------------------------------------------------------

REPAIR_TEMPLATE = (
    "Your previous response violated these constraints:\n{violations}\n\n"
    "Return a corrected JSON response. Same format."
)


def translate(
    lattice: dict[str, Any],
    model: str = DEFAULT_MODEL,
    repair: bool = True,
) -> dict[str, Any]:
    """Lattice in, validated translation out.

    One repair round only. A model that broke a constraint twice with the
    violations in front of it is not going to be talked round on a third try,
    and each round is a full request.
    """
    user_message = build_user_message(lattice)
    raw = call_llm(user_message, model)
    result = parse_response(raw)
    violations = validate(result, lattice)
    if not violations:
        return result

    if not repair:
        raise ValidationError(violations, raw)

    print(f"[slt] {len(violations)} violation(s), requesting repair",
          file=sys.stderr)
    followup = [
        {"role": "assistant", "content": raw},
        {"role": "user",
         "content": REPAIR_TEMPLATE.format(
             violations="\n".join(f"- {x}" for x in violations))},
    ]
    raw2 = call_llm(user_message, model, extra_messages=followup)
    result2 = parse_response(raw2)
    violations2 = validate(result2, lattice)
    if violations2:
        raise ValidationError(violations2, raw2)
    return result2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_result(result: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("GLOSS  " + " ".join(result.get("gloss_sequence") or []))
    out.append("")
    for i, tr in enumerate(result.get("translations") or [], 1):
        added = tr.get("added_words") or []
        tail = f"  (added: {', '.join(added)})" if added else ""
        out.append(f"{i}. {tr.get('text','')}{tail}")
        out.append(f"   [{tr.get('reading','?')}, confidence {tr.get('confidence','?')}]")
    notes = result.get("ambiguity_notes") or []
    if notes:
        out.append("")
        out.append("Ambiguity:")
        out.extend(f"  - {n}" for n in notes)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lattice", help="path to a lattice JSON file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--raw", action="store_true", help="print the JSON response")
    ap.add_argument("--no-repair", action="store_true",
                    help="fail on the first invalid response instead of "
                         "asking the model to correct it")
    args = ap.parse_args()

    try:
        lattice = load_lattice(args.lattice)
    except ValueError as exc:
        print(f"invalid lattice: {exc}", file=sys.stderr)
        return 2

    try:
        result = translate(lattice, args.model, repair=not args.no_repair)
    except ParseError as exc:
        print(f"could not parse response: {exc}", file=sys.stderr)
        return 3
    except ValidationError as exc:
        print("response failed validation:", file=sys.stderr)
        for x in exc.violations:
            print(f"  - {x}", file=sys.stderr)
        print("\n--- raw response ---", file=sys.stderr)
        print(exc.raw, file=sys.stderr)
        return 4
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 5

    print(json.dumps(result, ensure_ascii=False, indent=2) if args.raw
          else format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
