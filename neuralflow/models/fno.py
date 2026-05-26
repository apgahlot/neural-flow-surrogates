"""2-D Fourier Neural Operator (Li et al., 2021).

Deterministic autoregressive surrogate: maps conditioning channels
[permeability, S_t] (+ a coordinate grid) to the next saturation state S_{t+1}.
Uncertainty is obtained externally via a deep ensemble of independently
trained FNOs.
"""
from __future__ import annotations
import math

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
        # Real-valued (..., 2) storage; re-viewed as complex on demand.
        # Complex Parameters break GradScaler.unscale_ under AMP.
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))

    @staticmethod
    def _mul(x, w):
        return torch.einsum("bixy,ioxy->boxy", x, torch.view_as_complex(w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT path must run in float32 / complex64: ComplexHalf (from AMP fp16
        # rfft2) is unsupported by most ops, including einsum used below.
        in_dtype = x.dtype
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            b, _, h, w = x.shape
            m1 = min(self.modes1, h // 2)
            m2 = min(self.modes2, w // 2 + 1)
            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros(b, self.out_ch, h, w // 2 + 1,
                                 dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
            out_ft[:, :, -m1:, :m2] = self._mul(x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
            out = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return out.to(in_dtype)


class FNO2d(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 1, width: int = 36,
                 modes=(32, 32), n_layers: int = 4, residual=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.lift = nn.Conv2d(in_ch + 2, width, 1)  # +2 coordinate channels
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes[0], modes[1]) for _ in range(n_layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        ng = max(1, math.gcd(8, width))   # robust to widths not divisible by 8
        self.norms = nn.ModuleList([nn.GroupNorm(ng, width) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_ch, 1))

        # Per-channel residual mask (1 = predict delta + state_t, 0 = predict full).
        # For sat (sharp advancing front) residual prior fights the motion -> off.
        # For pres (smooth, slowly drifting) residual prior is correct -> on.
        if isinstance(residual, (list, tuple)):
            if len(residual) != out_ch:
                raise ValueError(
                    f"residual list length {len(residual)} != out_ch {out_ch}")
            mask = [1.0 if bool(r) else 0.0 for r in residual]
        else:
            mask = [1.0 if bool(residual) else 0.0] * out_ch
        self.residual_any = any(m > 0.5 for m in mask)
        self.register_buffer(
            "residual_mask", torch.tensor(mask).view(1, out_ch, 1, 1))

    @staticmethod
    def _coord_grid(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        gy = torch.linspace(0, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        gx = torch.linspace(0, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([gy, gx], dim=1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, in_ch, H, W) -> prediction (B, out_ch, H, W).

        cond is expected to be [perm, state_0, state_1, ...]; the residual
        branch uses cond[:, 1:1+out_ch] as the previous state.
        """
        x = torch.cat([cond, self._coord_grid(cond)], dim=1)
        x = self.lift(x)
        for sp, lc, nm in zip(self.spectral, self.local, self.norms):
            x = x + F.gelu(nm(sp(x) + lc(x)))
        delta = self.proj(x)
        if self.residual_any:
            state_t = cond[:, 1:1 + self.out_ch]
            return delta + state_t * self.residual_mask
        return delta
