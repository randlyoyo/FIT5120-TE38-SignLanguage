"""Guidance in one batch (text2sign.guided_batched) must equal the two-pass guidance it replaces.

The model here mimics SignSparK's UNet where it matters: the text embedding goes through
mask_cond(cond, force_mask=y["uncond"]) and is added to a per-sample time embedding; the rest is a
per-sample nonlinear function."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from signchat.text2sign import guided_batched  # noqa: E402


class FakeUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed_text = torch.nn.Linear(8, 6)
        self.vocab = {}

    def mask_cond(self, cond, force_mask=False):
        return torch.zeros_like(cond) if force_mask else cond

    def encode_text(self, texts):
        return torch.stack([self.vocab.setdefault(t, torch.randn(8)) for t in texts])

    def forward(self, x, t, y=None, obs_x0=None, obs_mask=None):
        emb = self.embed_text(self.mask_cond(self.encode_text(y["text"]), force_mask=y.get("uncond", False)))
        x = obs_x0 * obs_mask + x * (~obs_mask)
        return torch.tanh(x * (1 + t[:, None, None]) + emb[:, :, None]) * y["mask"]


def test_batched_guidance_equals_two_passes():
    model = FakeUNet()
    B, C, T = 3, 6, 10
    x, t = torch.randn(B, C, T), torch.rand(B)
    y = {"text": ["a .", "b .", "a ."], "mask": torch.ones(B, 1, T), "lengths": torch.tensor([10, 9, 8]),
         "keyframes": [[], [], []], "video_names": ["x", "y", "z"]}
    none = torch.zeros(B, C, T, dtype=torch.bool)
    fwd = lambda xx, tt, **kw: model(xx, tt, **kw)
    o = fwd(x, t, y=y, obs_x0=x, obs_mask=none)
    u = fwd(x, t, y=dict(y, uncond=True), obs_x0=x, obs_mask=none)
    ref = u + 2.5 * (o - u)
    got = guided_batched(model, fwd, 2.5)(x, t, y=y, obs_x0=x, obs_mask=none)
    assert torch.allclose(got, ref, atol=1e-6)
    assert "mask_cond" not in model.__dict__                       # the class method is back
    assert torch.equal(model.mask_cond(torch.ones(2, 8)), torch.ones(2, 8))
