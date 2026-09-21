"""Progressive-Transformer-style continuous pose decoder on pretrained mT5 blocks.

Progressive Transformers (Saunders et al. 2020) generate a pose sequence
autoregressively: each step consumes the previous frame plus a progress
counter in [0, 1], and emits the next frame plus its counter; generation stops
when the counter reaches ~1. Here the transformer layers are the 12 mT5-base
decoder blocks loaded from the Uni-Sign checkpoint (causal self-attention with
T5 relative position bias + cross-attention to the text encoder). Only the
token embedding and the vocabulary head are replaced:

    pose_in     Linear(J*C + 1 -> d_model)   previous frame (standardised) + counter
    pose_out    Linear(d_model -> J*C)       next frame (standardised)
    counter_out Linear(d_model -> 1)         next counter, sigmoid

Generation re-runs the decoder over the whole prefix each step instead of using
a KV cache. That is O(T^2) but T <= a few hundred frames, and it avoids
depending on transformers' Cache API, which changed between 4.x and 5.x.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import MT5Config
from transformers.models.mt5.modeling_mt5 import MT5Stack


def build_stack(config: MT5Config, is_decoder: bool) -> MT5Stack:
    cfg = MT5Config.from_dict(config.to_dict())
    cfg.is_decoder = is_decoder
    cfg.is_encoder_decoder = False
    cfg.use_cache = False
    if not is_decoder:
        cfg.num_layers = config.num_layers
    else:
        cfg.num_layers = config.num_decoder_layers
    # The stack's own embedding is never used (we pass inputs_embeds). In
    # transformers 5.x MT5Stack allocates one from vocab_size, so shrink it
    # to avoid a dead 192M-parameter matrix, then drop it.
    cfg.vocab_size = 1
    stack = MT5Stack(cfg)
    stack.embed_tokens = None
    return stack


def _stack_forward(stack: MT5Stack, inputs_embeds, attention_mask, encoder_hidden_states=None,
                   encoder_attention_mask=None) -> torch.Tensor:
    kwargs = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
    if encoder_hidden_states is not None:
        kwargs.update(encoder_hidden_states=encoder_hidden_states,
                      encoder_attention_mask=encoder_attention_mask)
    out = stack(**kwargs)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


class PoseDecoder(nn.Module):
    def __init__(self, config: MT5Config, num_joints: int, coord_dim: int):
        super().__init__()
        self.num_joints, self.coord_dim = num_joints, coord_dim
        d = config.d_model
        self.frame_dim = num_joints * coord_dim
        self.stack = build_stack(config, is_decoder=True)
        self.pose_in = nn.Linear(self.frame_dim + 1, d)
        self.pose_out = nn.Linear(d, self.frame_dim)
        self.counter_out = nn.Linear(d, 1)
        self.reset_new_parameters()

    @torch.no_grad()
    def reset_new_parameters(self, target_input_rms: float | None = None) -> None:
        """Initialise the replaced layers.

        pose_in is scaled so its outputs have roughly the RMS of the pretrained
        token embeddings the decoder blocks were trained on (inputs are ~unit
        variance after standardisation). pose_out starts near zero, i.e. near
        the mean pose, so early training does not fling the skeleton around.
        """
        fan_in = self.frame_dim + 1
        std_in = (target_input_rms or 1.0) / math.sqrt(fan_in)
        nn.init.normal_(self.pose_in.weight, std=std_in)
        nn.init.zeros_(self.pose_in.bias)
        nn.init.normal_(self.pose_out.weight, std=1e-3)
        nn.init.zeros_(self.pose_out.bias)
        nn.init.normal_(self.counter_out.weight, std=1e-3)
        nn.init.zeros_(self.counter_out.bias)

    def forward(self, prev_frames: torch.Tensor, prev_counter: torch.Tensor, frame_mask: torch.Tensor,
                enc_hidden: torch.Tensor, enc_mask: torch.Tensor):
        """prev_frames (B,T,J,C) standardised, prev_counter (B,T), frame_mask (B,T).

        Returns (frames (B,T,J,C) standardised, counter (B,T) in (0,1)).
        """
        B, T, J, C = prev_frames.shape
        if (J, C) != (self.num_joints, self.coord_dim):
            raise ValueError(f"expected (*,*,{self.num_joints},{self.coord_dim}), got {tuple(prev_frames.shape)}")
        x = torch.cat([prev_frames.reshape(B, T, J * C), prev_counter[..., None]], dim=-1)
        h = self.pose_in(x)
        h = _stack_forward(self.stack, h, frame_mask.long(), enc_hidden, enc_mask.long())
        frames = self.pose_out(h).view(B, T, J, C)
        counter = torch.sigmoid(self.counter_out(h).squeeze(-1))
        return frames, counter
