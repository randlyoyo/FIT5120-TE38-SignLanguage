#!/usr/bin/env python3
"""SLT evaluation: BLEU-1..4, ROUGE-L, BLEURT-20, and the OOV rate.

Auslan-Daily's two subsets are reported separately and there is no code path
here that averages them. Communication is multi-signer in-the-wild dialogue and
News is a single studio presenter; a mean over the two is a number that
describes neither, and it hides the case where the model only ever learned the
newsreader. If you need one figure for a slide, quote both.

BLEU is computed with sacrebleu so the numbers are comparable to the published
Uni-Sign / YouTube-SL-25 figures. ROUGE-L is implemented here rather than
pulled from a package because the third-party implementations disagree on
stemming and sentence splitting, and a metric that shifts when a dependency
updates is not a metric.

Expect single digits. YouTube-SL-25 reaches 15.4 BLEU on How2Sign with 1394
hours of ASL pre-training; this project has 45 hours of continuous Auslan and a
base model that saw no Auslan at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import oov_report, tokenise  # noqa: E402


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hyps: list[str], refs: list[str], beta: float = 1.2) -> float:
    """Corpus ROUGE-L as the mean of per-sentence F-measures."""
    scores = []
    for h, r in zip(hyps, refs):
        ht, rt = tokenise(h), tokenise(r)
        l = _lcs(ht, rt)
        if l == 0:
            scores.append(0.0)
            continue
        p, rc = l / len(ht), l / len(rt)
        scores.append(((1 + beta**2) * p * rc) / (rc + beta**2 * p))
    return 100.0 * sum(scores) / max(len(scores), 1)


def bleu_n(hyps: list[str], refs: list[str]) -> dict[str, float]:
    try:
        import sacrebleu
    except ImportError:
        raise SystemExit(
            "sacrebleu is required for comparable BLEU. pip install sacrebleu"
        )
    # Corpus BLEU at each order, sacrebleu defaults otherwise, so the figures
    # line up with what the sign-translation literature reports. No
    # effective_order: if the model produces no matching 4-gram anywhere in the
    # test set, BLEU-4 is 0 and that is the honest number.
    out = {}
    for n in (1, 2, 3, 4):
        metric = sacrebleu.metrics.BLEU(max_ngram_order=n)
        out[f"BLEU-{n}"] = round(metric.corpus_score(hyps, [refs]).score, 2)
    return out


def bleurt20(hyps: list[str], refs: list[str], batch_size: int = 16) -> float | None:
    """BLEURT-20 via the PyTorch port. Optional -- returns None if unavailable.

    Reported because it correlates with human judgement far better than BLEU at
    the low-BLEU end this project will live in, where a two-point BLEU
    difference is mostly noise.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None
    name = "lucadiliello/BLEURT-20"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    except Exception as exc:
        print(f"  (BLEURT-20 unavailable: {exc})", file=sys.stderr)
        return None
    scores = []
    with torch.no_grad():
        for i in range(0, len(hyps), batch_size):
            enc = tok(refs[i : i + batch_size], hyps[i : i + batch_size],
                      padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
            scores += model(**enc).logits.flatten().tolist()
    return round(100.0 * sum(scores) / max(len(scores), 1), 2)


def output_stats(hyps: list[str], refs: list[str]) -> dict:
    """How the outputs behave, whether or not they are right.

    An under-trained decoder falls back on a few stock sentences and on loops
    ("a bit of a bit of ..."), and BLEU shows neither. unique_hyps is the share
    of distinct outputs, looping the share containing a word trigram twice.
    """
    def loops(toks: list[str]) -> bool:
        grams = list(zip(toks, toks[1:], toks[2:]))
        return len(grams) != len(set(grams))

    toks = [tokenise(h) for h in hyps]
    n = max(len(hyps), 1)
    return {"unique_hyps": round(100.0 * len(set(hyps)) / n, 1),
            "looping": round(100.0 * sum(loops(t) for t in toks) / n, 1),
            "hyp_len": round(sum(map(len, toks)) / n, 1),
            "ref_len": round(sum(len(tokenise(r)) for r in refs) / n, 1)}


def score_group(hyps: list[str], refs: list[str], with_bleurt: bool) -> dict:
    m = bleu_n(hyps, refs)
    m["ROUGE-L"] = round(rouge_l(hyps, refs), 2)
    m.update(output_stats(hyps, refs))
    m["n"] = len(hyps)
    if with_bleurt:
        b = bleurt20(hyps, refs)
        if b is not None:
            m["BLEURT-20"] = b
    return m


def evaluate_records(records: list[dict], *, with_bleurt: bool = False,
                     train_texts: list[str] | None = None) -> dict:
    """records: [{uid, dataset, subset, ref, hyp}, ...] -> per-group metrics.

    Grouping is by dataset/subset and is not optional. There is deliberately no
    'overall' key.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[f"{r.get('dataset','?')}/{r.get('subset','?')}"].append(r)

    out: dict = {}
    for key, rows in sorted(groups.items()):
        hyps = [r["hyp"] for r in rows]
        refs = [r["ref"] for r in rows]
        out[key] = score_group(hyps, refs, with_bleurt)
        if train_texts is not None:
            out[key]["oov"] = oov_report(train_texts, refs)
    return out


def format_report(metrics: dict) -> str:
    lines = []
    for key, m in metrics.items():
        lines.append(f"\n{key}   (n={m['n']})")
        row = "  " + "  ".join(
            f"{k}={m[k]}" for k in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
                                    "ROUGE-L", "BLEURT-20") if k in m
        )
        lines.append(row)
        if "unique_hyps" in m:
            lines.append(
                f"  outputs: {m['unique_hyps']}% distinct, {m['looping']}% looping, "
                f"{m['hyp_len']} words on average vs {m['ref_len']} in the references"
            )
        if "oov" in m:
            o = m["oov"]
            lines.append(
                f"  OOV vs train: types {o['oov_type_rate']:.1%}, "
                f"tokens {o['oov_token_rate']:.1%}  "
                f"({o['oov_types']}/{o['test_types']} word types unseen)"
            )
            if o["examples"]:
                lines.append(f"    most frequent unseen: {', '.join(o['examples'][:10])}")
    lines.append("\nSubsets are reported separately by design; do not average them.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", type=Path, required=True,
                    help="jsonl with uid/dataset/subset/ref/hyp")
    ap.add_argument("--train-manifest", type=Path, default=None,
                    help="manifest of the training split, for the OOV rate")
    ap.add_argument("--bleurt", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    records = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
    train_texts = None
    if args.train_manifest:
        from manifest import read_manifest
        train_texts = [r["text"] for r in read_manifest(args.train_manifest)]

    metrics = evaluate_records(records, with_bleurt=args.bleurt,
                               train_texts=train_texts)
    print(format_report(metrics))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2))
        print(f"\n-> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
