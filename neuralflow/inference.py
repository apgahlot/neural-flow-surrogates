"""Autoregressive rollout for the joint-state flow surrogates.

All rollout functions take a normalized permeability field (H, W) and a
normalized initial joint state (2, H, W) = [S_0, P_0], and return a normalized
trajectory of shape (n_members_or_samples, n_steps + 1, 2, H, W).  Callers
denormalize each channel separately before computing physical-unit metrics.
"""
from __future__ import annotations

import numpy as np
import torch


def _cond(perm: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Build the (1, 3, H, W) conditioning batch from perm (H,W) and state (2,H,W)."""
    return torch.cat([perm.unsqueeze(0), state], dim=0).unsqueeze(0)


@torch.no_grad()
def rollout_fno(models, perm, state0, n_steps: int, device) -> np.ndarray:
    """Deep-ensemble FNO rollout over the joint (sat, pres) state.

    models  : list of FNO2d modules (ensemble members)
    perm    : (H, W) normalized permeability tensor
    state0  : (2, H, W) normalized initial state [S_0, P_0]
    returns : (M, n_steps + 1, 2, H, W) normalized trajectories
    """
    h, w = perm.shape
    perm = perm.to(device)
    out = np.zeros((len(models), n_steps + 1, 2, h, w), dtype=np.float32)
    for m, model in enumerate(models):
        model.eval().to(device)
        s = state0.to(device)
        out[m, 0] = s.cpu().numpy()
        for k in range(n_steps):
            s = model(_cond(perm, s))[0].clamp(-1, 1)         # (2, H, W)
            out[m, k + 1] = s.cpu().numpy()
    return out


@torch.no_grad()
def rollout_diffusion(diffusion, perm, state0, n_steps: int, n_samples: int,
                      device, sampler: str = "ddim", steps: int = 50) -> np.ndarray:
    """Generative rollout over the joint state: `n_samples` independent trajectories.

    returns : (n_samples, n_steps + 1, 2, H, W) normalized trajectories
    """
    h, w = perm.shape
    perm = perm.to(device)
    diffusion.eval().to(device)
    out = np.zeros((n_samples, n_steps + 1, 2, h, w), dtype=np.float32)
    for n in range(n_samples):
        s = state0.to(device)
        out[n, 0] = s.cpu().numpy()
        for k in range(n_steps):
            cond = _cond(perm, s)                              # (1, 3, H, W)
            shape = (1, 2, h, w)
            s = diffusion.sample(cond, shape=shape, sampler=sampler, steps=steps)[0].clamp(-1, 1)
            out[n, k + 1] = s.cpu().numpy()
    return out


def ensemble_stats(traj: np.ndarray):
    """Return (mean, std) over the ensemble/sample axis (axis 0)."""
    return traj.mean(axis=0), traj.std(axis=0)
