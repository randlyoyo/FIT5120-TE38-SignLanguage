#!/usr/bin/env python3
"""A small, real, self-contained SLT model used to test the harness itself.

This is NOT the research model and must never appear in a reported result. Its
job is to let the whole pipeline -- dataloading, the four-part input plumbing,
the Arm B freezing policy, the Arm C loss mixing, checkpointing, decoding,
evaluation -- be executed and debugged before the Uni-Sign checkpoint is in
hand. Harness bugs found this way are found for free; found later they are
found while holding a GPU and a half-finished experiment.

Module names deliberately mirror what the group patterns in model_adapter.py
expect, so the freezing policy is exercised for real rather than mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class WordVocab:
    def __init__(self, texts: list[str], min_count: int = 1):
        from collections import Counter

        from dataset import tokenise

        c = Counter(w for t in texts for w in tokenise(t))
        self.itos = SPECIALS + [w for w, n in c.most_common() if n >= min_count]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int = 48) -> list[int]:
        from dataset import tokenise

        ids = [self.stoi.get(w, UNK) for w in tokenise(text)][: max_len - 2]
        return [BOS] + ids + [EOS]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == EOS:
                break
            if i in (PAD, BOS):
                continue
            out.append(self.itos[i] if i < len(self.itos) else "<unk>")
        return " ".join(out)


def _causal_mask(n: int, device) -> torch.Tensor:
    """Boolean causal mask. Bool to match the boolean padding masks -- torch
    deprecates mixing a float attn_mask with a bool key_padding_mask."""
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


class _PartEncoder(nn.Module):
    def __init__(self, n_joints: int, d: int):
        super().__init__()
        # 3 channels per joint: x, y, confidence -- as in Uni-Sign's
        # proj_linear = nn.Linear(3, 64).
        self.net = nn.Sequential(nn.Linear(n_joints * 3, d), nn.GELU(),
                                 nn.Linear(d, d))

    def forward(self, x):                      # (B, T, J, 3)
        B, T = x.shape[:2]
        return self.net(x.reshape(B, T, -1))


class SmokeSLT(nn.Module):
    """Four per-part encoders -> temporal transformer -> transformer decoder."""

    def __init__(self, vocab_size: int, d: int = 128, nhead: int = 4,
                 enc_layers: int = 2, dec_layers: int = 2, max_out: int = 48):
        super().__init__()
        self.d = d
        self.max_out = max_out
        # Four branches, THREE sets of weights: Uni-Sign aliases the left hand
        # encoder to the right one (`gcn_modules['left'] = gcn_modules['right']`).
        # Mirrored here so the freezing policy and the parameter counts see the
        # same topology they will see on the real model.
        self.proj_linear = nn.ModuleDict({
            "body": _PartEncoder(spec.PART_SIZES["body"], d),
            "right": _PartEncoder(spec.PART_SIZES["right"], d),
            "face_all": _PartEncoder(spec.PART_SIZES["face_all"], d),
        })
        self.proj_linear["left"] = self.proj_linear["right"]

        self.pose_proj = nn.Linear(4 * d, d)
        self.temporal_pos = nn.Parameter(torch.zeros(1, 4096, d))
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, 4 * d, batch_first=True,
                                       norm_first=True), enc_layers)

        self.decoder_embed = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.decoder_pos = nn.Parameter(torch.zeros(1, max_out, d))
        self.decoder_stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, nhead, 4 * d, batch_first=True,
                                       norm_first=True), dec_layers)
        self.lm_head = nn.Linear(d, vocab_size)

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        feats = [self.proj_linear[p](batch[p]) for p in spec.PART_ORDER]
        h = self.pose_proj(torch.cat(feats, dim=-1))
        T = h.shape[1]
        h = h + self.temporal_pos[:, :T]
        pad = batch["attention_mask"] == 0
        # A clip where every frame was masked out would make the whole row pad
        # and produce NaN attention; keep the first frame alive in that case.
        pad[pad.all(dim=1), 0] = False
        return self.temporal_encoder(h, src_key_padding_mask=pad), pad

    def forward(self, batch: dict, tgt: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits over tgt[:, 1:]."""
        mem, mem_pad = self.encode(batch)
        inp = tgt[:, :-1]
        y = self.decoder_embed(inp) + self.decoder_pos[:, : inp.shape[1]]
        causal = _causal_mask(inp.shape[1], inp.device)
        out = self.decoder_stack(y, mem, tgt_mask=causal,
                                 tgt_key_padding_mask=inp.eq(PAD),
                                 memory_key_padding_mask=mem_pad)
        return self.lm_head(out)

    @torch.no_grad()
    def generate(self, batch: dict, max_len: int | None = None) -> list[list[int]]:
        mem, mem_pad = self.encode(batch)
        B = mem.shape[0]
        max_len = min(max_len or self.max_out, self.max_out)
        ids = torch.full((B, 1), BOS, dtype=torch.long, device=mem.device)
        done = torch.zeros(B, dtype=torch.bool, device=mem.device)
        for _ in range(max_len - 1):
            y = self.decoder_embed(ids) + self.decoder_pos[:, : ids.shape[1]]
            causal = _causal_mask(ids.shape[1], ids.device)
            out = self.decoder_stack(y, mem, tgt_mask=causal,
                                     memory_key_padding_mask=mem_pad)
            nxt = self.lm_head(out[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            done |= nxt.eq(EOS)
            if bool(done.all()):
                break
        return ids.tolist()


def pad_targets(seqs: list[list[int]], device) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    out = torch.full((len(seqs), n), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out.to(device)
