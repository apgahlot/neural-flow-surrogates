"""Conditional U-Net denoiser for the diffusion surrogate.

Input  : noisy target S_{t+1} (1 ch) concatenated with conditioning
         [permeability, S_t] (2 ch), plus a diffusion-timestep embedding.
Output : predicted noise (1 ch).

Self-attention is applied only at the bottleneck — at 256x512 inputs, attention
at higher resolutions is prohibitively memory-hungry.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of diffusion timesteps."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(c), dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class CondUNet(nn.Module):
    """U-Net denoiser. forward(x_noisy, t, cond) -> predicted noise."""

    def __init__(self, target_ch: int = 1, cond_ch: int = 2, base: int = 64,
                 channel_mults=(1, 2, 2, 4)):
        super().__init__()
        t_dim = base * 4
        self.base = base
        self.t_mlp = nn.Sequential(
            nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.in_conv = nn.Conv2d(target_ch + cond_ch, base, 3, padding=1)

        n = len(channel_mults)
        self.enc_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        chs = [base]
        ch = base
        for i, mult in enumerate(channel_mults):
            out = base * mult
            self.enc_blocks.append(ResBlock(ch, out, t_dim))
            ch = out
            chs.append(ch)
            if i < n - 1:
                self.downsamplers.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                chs.append(ch)
            else:
                self.downsamplers.append(None)

        self.mid1 = ResBlock(ch, ch, t_dim)
        self.mid_attn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, t_dim)

        self.dec_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out = base * mult
            blocks = nn.ModuleList()
            for _ in range(2):
                blocks.append(ResBlock(ch + chs.pop(), out, t_dim))
                ch = out
            self.dec_blocks.append(blocks)
            self.upsamplers.append(
                nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1) if i > 0 else None)

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, target_ch, 3, padding=1))

    def forward(self, x_noisy, t, cond):
        t_emb = self.t_mlp(timestep_embedding(t, self.base))
        h = self.in_conv(torch.cat([x_noisy, cond], dim=1))
        skips = [h]
        for blk, down in zip(self.enc_blocks, self.downsamplers):
            h = blk(h, t_emb)
            skips.append(h)
            if down is not None:
                h = down(h)
                skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)
        for blocks, up in zip(self.dec_blocks, self.upsamplers):
            for blk in blocks:
                h = blk(torch.cat([h, skips.pop()], dim=1), t_emb)
            if up is not None:
                h = up(h)
        return self.out(h)
