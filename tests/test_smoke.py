"""Smoke tests: model shapes, gradients, and metrics on tiny synthetic inputs.

These run on CPU in seconds and need no data — they guard against shape and
wiring regressions.
"""
import numpy as np
import torch

from neuralflow import metrics
from neuralflow.models.diffusion import GaussianDiffusion
from neuralflow.models.dit import DiT
from neuralflow.models.fno import FNO2d
from neuralflow.models.unet import CondUNet

H, W = 64, 32          # tiny stand-in for the real 256 x 512 grid
TC, CC = 2, 3          # target channels (sat, pres); cond channels (perm, S_t, P_t)


def test_fno_forward_backward():
    model = FNO2d(in_ch=CC, out_ch=TC, width=8, modes=(8, 8), n_layers=2)
    cond = torch.randn(2, CC, H, W)
    out = model(cond)
    assert out.shape == (2, TC, H, W)
    out.pow(2).mean().backward()


def test_unet_diffusion():
    denoiser = CondUNet(target_ch=TC, cond_ch=CC, base=16, channel_mults=(1, 2, 2))
    diff = GaussianDiffusion(denoiser, timesteps=20)
    x0 = torch.randn(2, TC, H, W)
    cond = torch.randn(2, CC, H, W)
    loss = diff.loss(x0, cond)
    loss.backward()
    sample = diff.sample(cond, shape=(2, TC, H, W), sampler="ddim", steps=3)
    assert sample.shape == (2, TC, H, W)


def test_dit_diffusion():
    denoiser = DiT(height=H, width=W, target_ch=TC, cond_ch=CC,
                   patch_size=8, hidden=32, depth=2, heads=4)
    diff = GaussianDiffusion(denoiser, timesteps=20)
    x0 = torch.randn(2, TC, H, W)
    cond = torch.randn(2, CC, H, W)
    diff.loss(x0, cond).backward()
    assert diff.sample(cond, shape=(2, TC, H, W), sampler="ddim", steps=3).shape == (2, TC, H, W)


def test_metrics():
    true = np.random.rand(4, H, W).astype("float32")
    samples = np.random.rand(8, 4, H, W).astype("float32")
    assert metrics.relative_l2(samples[0], true) >= 0.0
    assert 0.0 <= metrics.plume_iou(samples[0], true) <= 1.0
    assert np.isfinite(metrics.crps_ensemble(samples, true))
    cov = metrics.interval_coverage(samples, true)
    assert all(0.0 <= v <= 1.0 for v in cov.values())
    nominal, empirical = metrics.reliability_curve(samples, true)
    assert len(nominal) == len(empirical)
