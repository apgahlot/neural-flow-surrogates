"""Autoregressive rollout for the flow surrogates.

All rollout functions take a normalized permeability field and a normalized
initial saturation state, and return a normalized trajectory of shape
(n_members_or_samples, n_steps + 1, H, W). Callers denormalize with
`Normalizer.denorm_sat` before computing physical-unit metrics.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def rollout_fno(models, perm, sat0, n_steps: int, device) -> np.ndarray:
    """Deep-ensemble FNO rollout.

    models : list of FNO2d modules (ensemble members)
    perm   : (H, W) normalized permeability tensor
    sat0   : (H, W) normalized initial saturation tensor
    returns: (M, n_steps + 1, H, W) normalized trajectories
    """
    h, w = perm.shape
    perm = perm.to(device)
    out = np.zeros((len(models), n_steps + 1, h, w), dtype=np.float32)
    for m, model in enumerate(models):
        model.eval().to(device)
        s = sat0.to(device)
        out[m, 0] = s.cpu().numpy()
        for k in range(n_steps):
            cond = torch.stack([perm, s], dim=0).unsqueeze(0)   # (1, 2, H, W)
            s = model(cond)[0, 0].clamp(-1, 1)
            out[m, k + 1] = s.cpu().numpy()
    return out


@torch.no_grad()
def rollout_diffusion(diffusion, perm, sat0, n_steps: int, n_samples: int,
                      device, sampler: str = "ddim", steps: int = 50) -> np.ndarray:
    """Generative rollout: `n_samples` independent autoregressive trajectories.

    returns: (n_samples, n_steps + 1, H, W) normalized trajectories
    """
    h, w = perm.shape
    perm = perm.to(device)
    diffusion.eval().to(device)
    out = np.zeros((n_samples, n_steps + 1, h, w), dtype=np.float32)
    for n in range(n_samples):
        s = sat0.to(device)
        out[n, 0] = s.cpu().numpy()
        for k in range(n_steps):
            cond = torch.stack([perm, s], dim=0).unsqueeze(0)   # (1, 2, H, W)
            s = diffusion.sample(cond, sampler=sampler, steps=steps)[0, 0].clamp(-1, 1)
            out[n, k + 1] = s.cpu().numpy()
    return out


def ensemble_stats(traj: np.ndarray):
    """Return (mean, std) over the ensemble/sample axis (axis 0)."""
    return traj.mean(axis=0), traj.std(axis=0)
