"""Back-translation evaluation interface: generated pose -> SLR/SLT model -> English.

    generated pose (T, J, 2)  ->  SLRBackTranslator.translate  ->  text  ->  BLEU / ROUGE vs input

Any recogniser can plug in by implementing `translate`. The provided
implementation drives a Uni-Sign pose-only SLT model (e.g. the Auslan-Daily
fine-tuned runs from ../unisign, whose Communication BLEU-4 on real poses is
the ceiling to compare against). It reuses ../unisign/model_adapter.py to build
the model from the Uni-Sign repo and converts poses with the same converter as
the feature loss, using the dataset's canonical camera.

Caveats that decide whether the number means anything:
  * The SLT model must NOT have been trained on the clips whose text is being
    generated. The ../unisign runs trained on Auslan-Daily's official train
    split: evaluate on official val/test only (split.method: column).
  * Run the same SLT model on the GROUND-TRUTH poses of the same clips and
    report both; generated-pose BLEU is only interpretable relative to that.
  * High back-translation BLEU can come from an SLT model keying on a few
    salient features; it is evidence of intelligibility to that model, not to
    Deaf signers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from slp_models import skeleton as sk
from slp_utils.metrics import resample_to
from slp_utils.unisign_format import pad_with_last_frame, to_unisign_parts


class SLRBackTranslator(Protocol):
    def translate(self, poses: list[np.ndarray], fps: float) -> list[str]: ...


class UniSignBackTranslator:
    def __init__(self, checkpoint: Path, unisign_repo: Path, mt5_dir: Path, canonical_camera: list[float],
                 mean_confidence: list[float], slt_fps: float, device: str = "cpu",
                 num_beams: int = 4, max_new_tokens: int = 100):
        here = Path(__file__).resolve().parents[2] / "unisign"
        if not (here / "model_adapter.py").is_file():
            raise FileNotFoundError(f"../unisign/model_adapter.py not found at {here}")
        if str(here) not in sys.path:
            sys.path.append(str(here))
        import model_adapter  # noqa: E402  (../unisign)
        self.model = model_adapter.load_unisign(checkpoint, unisign_repo, mt5_dir, device=device).eval()
        self.device = device
        self.cam = torch.tensor(canonical_camera, dtype=torch.float32)
        self.conf = torch.tensor(mean_confidence, dtype=torch.float32)
        self.slt_fps = slt_fps
        self.num_beams, self.max_new_tokens = num_beams, max_new_tokens

    @torch.no_grad()
    def translate(self, poses: list[np.ndarray], fps: float, batch_size: int = 8) -> list[str]:
        out: list[str] = []
        for i in range(0, len(poses), batch_size):
            chunk = []
            for p in poses[i:i + batch_size]:
                T = max(2, int(round(p.shape[0] * self.slt_fps / fps)))
                chunk.append(resample_to(np.nan_to_num(p), T).astype(np.float32))
            B, T = len(chunk), max(c.shape[0] for c in chunk)
            pose = torch.zeros(B, T, sk.NUM_JOINTS, 2)
            lengths = torch.tensor([c.shape[0] for c in chunk])
            for b, c in enumerate(chunk):
                pose[b, : c.shape[0]] = torch.from_numpy(c)
            pose = pad_with_last_frame(pose, lengths)
            valid = torch.ones(B, T, sk.NUM_JOINTS, dtype=torch.bool)
            parts = to_unisign_parts(pose, valid, self.cam.expand(B, -1), self.conf)
            src = {k: v.to(self.device) for k, v in parts.items()}
            am = (torch.arange(T)[None] < lengths[:, None]).long()
            src["attention_mask"] = am.to(self.device)
            stack = self.model(src, {"gt_sentence": ["."] * B})
            ids = self.model.generate(stack, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams)
            out.extend(self.model.mt5_tokenizer.batch_decode(ids, skip_special_tokens=True))
        return out


def text_scores(hyps: list[str], refs: list[str]) -> dict:
    try:
        import sacrebleu
    except ImportError:
        return {"error": "pip install sacrebleu for BLEU"}
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {"bleu4": bleu.score, "bleu_precisions": bleu.precisions, "chrf": chrf.score, "n": len(hyps)}
