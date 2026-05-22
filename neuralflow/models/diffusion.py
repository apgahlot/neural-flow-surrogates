"""Gaussian diffusion (DDPM) wrapper for a conditional denoiser.

The denoiser (U-Net or DiT) predicts the noise epsilon given a noisy target,
the diffusion timestep, and the spatial conditioning [permeability, S_t].
Uncertainty is obtained by drawing multiple samples (different initial noise).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(name: str, T: int) -> torch.Tensor:
    if name == "linear":
        return torch.linspace(1e-4, 0.02, T)
    # cosine schedule (Nichol & Dhariwal, 2021)
    s = 0.008
    x = torch.linspace(0, T, T + 1)
    acp = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - acp[1:] / acp[:-1]
    return betas.clamp(1e-8, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, denoiser: nn.Module, timesteps: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        betas = make_beta_schedule(schedule, timesteps)
        acp = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", acp.sqrt())
        self.register_buffer("sqrt_one_minus_acp", (1.0 - acp).sqrt())

    # -- training -----------------------------------------------------------
    def q_sample(self, x0, t, noise):
        a = self.sqrt_acp[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_acp[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Standard epsilon-prediction MSE loss."""
        bsz = x0.shape[0]
        t = torch.randint(0, self.timesteps, (bsz,), device=x0.device)
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        pred = self.denoiser(x_noisy, t, cond)
        return F.mse_loss(pred, noise)

    # -- sampling -----------------------------------------------------------
    @torch.no_grad()
    def ddim_sample(self, cond, shape, steps: int = 50, eta: float = 0.0):
        device = cond.device
        bsz = shape[0]
        times = torch.linspace(self.timesteps - 1, 0, steps + 1).round().long().to(device)
        x = torch.randn(shape, device=device)
        for i in range(steps):
            t, t_next = times[i], times[i + 1]
            t_b = torch.full((bsz,), int(t), device=device, dtype=torch.long)
            eps = self.denoiser(x, t_b, cond)
            a_t = self.alphas_cumprod[t]
            a_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            sigma = eta * (((1 - a_next) / (1 - a_t)) * (1 - a_t / a_next)).clamp(min=0).sqrt()
            x = a_next.sqrt() * x0 + (1 - a_next - sigma ** 2).clamp(min=0).sqrt() * eps
            if eta > 0 and i < steps - 1:
                x = x + sigma * torch.randn_like(x)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def ddpm_sample(self, cond, shape):
        device = cond.device
        bsz = shape[0]
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_b = torch.full((bsz,), t, device=device, dtype=torch.long)
            eps = self.denoiser(x, t_b, cond)
            beta = self.betas[t]
            a_t = self.alphas_cumprod[t]
            coef = beta / self.sqrt_one_minus_acp[t]
            mean = (x - coef * eps) / (1 - beta).sqrt()
            if t > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample(self, cond, shape=None, sampler: str = "ddim", steps: int = 50):
        """Generate one prediction per conditioning example."""
        if shape is None:
            shape = (cond.shape[0], 1, cond.shape[-2], cond.shape[-1])
        if sampler == "ddpm":
            return self.ddpm_sample(cond, shape)
        return self.ddim_sample(cond, shape, steps=steps)
