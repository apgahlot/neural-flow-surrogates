"""Diffusion Transformer (DiT) denoiser  --  Phase 2.

A transformer backbone for the diffusion surrogate, as an alternative to the
U-Net. The noisy target and the spatial conditioning [permeability, S_t] are
patchified together; the diffusion timestep conditions every block through
adaLN-Zero modulation (Peebles & Xie, 2023).

Note: DiTs are data-hungry. With ~128 simulations (~2.9k autoregressive pairs)
this is expected to trail the U-Net diffusion model; the sample-efficiency gap
is itself a reported result. Keep `hidden_size` / `depth` modest.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn

from .unet import timestep_embedding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift_a, scale_a)
        x = x + gate_a.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x


class DiT(nn.Module):
    """forward(x_noisy, t, cond) -> predicted noise. Signature matches CondUNet."""

    def __init__(self, height: int, width: int, target_ch: int = 1, cond_ch: int = 2,
                 patch_size: int = 16, hidden: int = 384, depth: int = 8, heads: int = 6):
        super().__init__()
        assert height % patch_size == 0 and width % patch_size == 0, \
            "height and width must be divisible by patch_size"
        self.patch = patch_size
        self.target_ch = target_ch
        self.gh, self.gw = height // patch_size, width // patch_size
        n_tokens = self.gh * self.gw
        in_ch = target_ch + cond_ch

        self.proj_in = nn.Conv2d(in_ch, hidden, patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, hidden))
        self.t_embed = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.proj_out = nn.Linear(hidden, patch_size * patch_size * target_ch)
        self.hidden = hidden
        nn.init.trunc_normal_(self.pos, std=0.02)

    def unpatchify(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.gh, self.gw, self.patch, self.patch, self.target_ch)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, self.target_ch, self.gh * self.patch, self.gw * self.patch)

    def forward(self, x_noisy, t, cond):
        x = torch.cat([x_noisy, cond], dim=1)
        x = self.proj_in(x).flatten(2).transpose(1, 2)        # (B, N, hidden)
        x = x + self.pos
        c = self.t_embed(timestep_embedding(t, self.hidden))  # (B, hidden)
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.ada_out(c).chunk(2, dim=1)
        x = modulate(self.norm_out(x), shift, scale)
        x = self.proj_out(x)                                  # (B, N, p*p*C)
        return self.unpatchify(x)
