#!/usr/bin/env python3
"""Decoding options and --eval-only.

  * UniSignBackend.generate makes exactly the call Uni_Sign.generate makes
    (the repo's own function, not a copy of it), plus the config's `decode`
    options
  * --eval-only scores a finished run into eval_<tag>/ with the run's trained
    weights, leaves its predictions, metrics and checkpoints alone, and refuses
    when there are no weights
  * a changed `decode` block does not stop --resume
  * evaluate.output_stats counts distinct and looping outputs

    python test_decode.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "Uni-Sign"
sys.path.insert(0, str(HERE))
import train as T  # noqa: E402
from evaluate import output_stats  # noqa: E402
from test_resume import make_corpus  # noqa: E402

failures = 0


def check(ok: bool, what: str) -> None:
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {what}")


# --- the generate call --------------------------------------------------------

class FakeMT5:
    def __init__(self):
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        return torch.tensor([[0, 7, 1]])


class FakeUniSign:
    """What UniSignBackend touches on the model: forward, mt5_model, the tokenizer."""
    def __init__(self):
        self.mt5_model = FakeMT5()
        self.mt5_tokenizer = type("Tok", (), {
            "batch_decode": staticmethod(lambda ids, skip_special_tokens: ["hyp"])})()

    def __call__(self, src, tgt):
        return {"inputs_embeds": torch.zeros(1, 3, 4), "attention_mask": torch.ones(1, 3),
                "loss": torch.tensor(0.0)}


def backend_with(decode: dict):
    b = T.UniSignBackend.__new__(T.UniSignBackend)       # no checkpoint needed
    b.model, b.max_new_tokens, b.num_beams, b.decode = FakeUniSign(), 100, 5, decode
    return b


def same_call(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


def test_generate_call() -> None:
    sys.path.insert(0, str(REPO))
    import models                                   # the Uni-Sign repo's own code

    batch = {"text": ["a sentence"]}
    plain = backend_with({})
    out = plain.model(None, None)
    models.Uni_Sign.generate(plain.model, out, max_new_tokens=100, num_beams=5)
    upstream = plain.model.mt5_model.calls.pop()
    plain.generate(batch)
    ours = plain.model.mt5_model.calls.pop()
    check(same_call(ours, upstream),
          "no decode options -> exactly Uni_Sign.generate's call to mT5")

    opts = {"no_repeat_ngram_size": 3, "repetition_penalty": 1.2}
    b = backend_with(opts)
    b.generate(batch)
    call = b.model.mt5_model.calls.pop()
    check(all(call.get(k) == v for k, v in opts.items())
          and same_call({k: v for k, v in call.items() if k not in opts}, upstream),
          "decode options -> passed to mT5's generate on top of the same call")


# --- --eval-only on a finished smoke run --------------------------------------

def cmd(data: Path, out: Path, *extra: str) -> list[str]:
    return [sys.executable, str(HERE / "train.py"), "--config", str(HERE / "configs" / "arm_a.yaml"),
            "--out", str(out), "--device", "cpu", *extra, "--set",
            f"data.manifest={data / 'manifest.jsonl'}", f"data.npz_dir={data / 'pose'}",
            "num_workers=0", "log_every=1000", "optim.epochs=2", "backend.d=32",
            "checkpoint.every_steps=5", "checkpoint.every_minutes=null", "checkpoint.keep=1"]


def run(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(list(a), capture_output=True, text=True)


def digest(paths) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def test_eval_only(tmp: Path) -> None:
    data, out = tmp / "data", tmp / "run"
    make_corpus(data)
    tr = run(*cmd(data, out))
    if tr.returncode != 0:
        print(tr.stdout[-1500:], tr.stderr[-1500:])
    check(tr.returncode == 0 and (out / "metrics.json").exists(), "a smoke run trains and evaluates")
    m = json.loads((out / "metrics.json").read_text())
    check(all("unique_hyps" in g and "looping" in g for g in m.values()),
          "metrics.json carries the output statistics")

    own = digest([out / "metrics.json", out / "predictions.jsonl",
                  *(out / "checkpoints").glob("*.pt"), out / "checkpoint.pt"])
    ev = run(*cmd(data, out, "--eval-only", "--eval-tag", "nr3",
                  "--set", "decode.no_repeat_ngram_size=3"))
    if ev.returncode != 0:
        print(ev.stdout[-1500:], ev.stderr[-1500:])
    dest = out / "eval_nr3"
    check(ev.returncode == 0 and all((dest / f).exists() for f in
                                     ("predictions.jsonl", "metrics.json", "settings.json")),
          "--eval-only on a finished run (checkpoints present, no --resume) writes eval_nr3/")
    settings = json.loads((dest / "settings.json").read_text())
    check(settings["decode"] == {"no_repeat_ngram_size": 3}
          and settings["weights"].endswith("checkpoint.pt"),
          "settings.json records the decoding and the weights")
    check("ignored: the smoke backend" in ev.stdout,
          "the smoke backend says it ignores decode options")
    check(digest([out / "metrics.json", out / "predictions.jsonl",
                  *(out / "checkpoints").glob("*.pt"), out / "checkpoint.pt"]) == own,
          "the run's own metrics, predictions and checkpoints are untouched")
    # Smoke decoding is greedy and ignores the options, so the trained weights
    # must give back the run's own predictions exactly; freshly initialised
    # weights would not.
    check((dest / "predictions.jsonl").read_text() == (out / "predictions.jsonl").read_text(),
          "--eval-only loads the trained weights (same predictions as the run's own)")

    missing = run(*cmd(data, tmp / "never_trained", "--eval-only"))
    check(missing.returncode != 0 and "not found" in missing.stderr + missing.stdout,
          "--eval-only without weights refuses")

    rs = run(*cmd(data, out, "--resume", "--set", "decode.no_repeat_ngram_size=4"))
    check(rs.returncode == 0 and "run already finished" in rs.stdout
          and "different run" not in rs.stdout + rs.stderr,
          "--resume with a changed decode block is not refused")


def test_output_stats() -> None:
    s = output_stats(["a bit of a bit of", "hello .", "hello ."], ["x y", "hello .", "hi ."])
    check(s["unique_hyps"] == 66.7 and s["looping"] == 33.3 and s["hyp_len"] == 2.7,
          f"output_stats on a toy set -> {s}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="test-decode-"))
    try:
        test_generate_call()
        test_output_stats()
        test_eval_only(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PASS" if failures == 0 else f"\n{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
