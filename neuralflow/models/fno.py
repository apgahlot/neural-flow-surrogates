"""2-D Fourier Neural Operator (Li et al., 2021).

Deterministic autoregressive surrogate: maps conditioning channels
[permeability, S_t] (+ a coordinate grid) to the next saturation state S_{t+1}.
Uncertainty is obtained externally via a deep ensemble of independently
trained FNOs.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Spectral convolution: linear transform on the lowest Fourier modes."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        m1 = min(self.modes1, h // 2)
        m2 = min(self.modes2, w // 2 + 1)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, self.out_ch, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._mul(x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 1, width: int = 36,
                 modes=(32, 32), n_layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch + 2, width, 1)  # +2 coordinate channels
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes[0], modes[1]) for _ in range(n_layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_ch, 1))

    @staticmethod
    def _coord_grid(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        gy = torch.linspace(0, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        gx = torch.linspace(0, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([gy, gx], dim=1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, in_ch, H, W) -> prediction (B, out_ch, H, W)."""
        x = torch.cat([cond, self._coord_grid(cond)], dim=1)
        x = self.lift(x)
        for sp, lc, nm in zip(self.spectral, self.local, self.norms):
            x = x + F.gelu(nm(sp(x) + lc(x)))
        return self.proj(x)
