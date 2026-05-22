"""Accuracy, uncertainty-calibration and front-fidelity metrics.

All functions accept NumPy arrays or Torch tensors and return plain floats /
NumPy arrays. Saturation fields are compared in physical [0, 1] units.
"""
from __future__ import annotations
import numpy as np


def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------
def relative_l2(pred, true, eps: float = 1e-8) -> float:
    """Mean over the batch of ||pred - true||_2 / ||true||_2."""
    pred, true = _np(pred), _np(true)
    b = pred.shape[0]
    p, t = pred.reshape(b, -1), true.reshape(b, -1)
    num = np.linalg.norm(p - t, axis=1)
    den = np.linalg.norm(t, axis=1) + eps
    return float(np.mean(num / den))


def rmse(pred, true) -> float:
    pred, true = _np(pred), _np(true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true) -> float:
    pred, true = _np(pred), _np(true)
    return float(np.mean(np.abs(pred - true)))


# --------------------------------------------------------------------------
# uncertainty calibration
# --------------------------------------------------------------------------
def crps_ensemble(samples, true) -> float:
    """CRPS for an ensemble / set of generative samples.

    samples : (M, ...) ensemble members
    true    : (...)    observation
    CRPS = E|X - y| - 0.5 E|X - X'|, estimated from the M samples and averaged
    over all elements. The pairwise term is accumulated in a loop to avoid
    materializing the (M, M, ...) difference tensor.
    """
    samples, true = _np(samples), _np(true)
    m = samples.shape[0]
    term1 = float(np.mean(np.abs(samples - true[None])))
    pair_sum = np.zeros(true.shape, dtype=np.float64)
    for i in range(m):
        pair_sum += np.abs(samples[i][None] - samples).sum(axis=0)
    term2 = float((pair_sum / (m * m)).mean())
    return term1 - 0.5 * term2


def interval_coverage(samples, true, levels=(0.5, 0.8, 0.9, 0.95)) -> dict:
    """Empirical coverage of central prediction intervals at nominal `levels`.

    samples : (M, ...)   true : (...)
    Returns {nominal_level: empirical_coverage}.
    """
    samples, true = _np(samples), _np(true)
    out = {}
    for lvl in levels:
        lo = np.quantile(samples, (1 - lvl) / 2, axis=0)
        hi = np.quantile(samples, 1 - (1 - lvl) / 2, axis=0)
        inside = (true >= lo) & (true <= hi)
        out[float(lvl)] = float(np.mean(inside))
    return out


def reliability_curve(samples, true, n_levels: int = 11):
    """Nominal vs. empirical coverage for a reliability diagram."""
    levels = np.linspace(0.0, 1.0, n_levels)[1:-1]
    cov = interval_coverage(samples, true, levels=tuple(levels))
    nominal = np.array(sorted(cov.keys()))
    empirical = np.array([cov[l] for l in nominal])
    return nominal, empirical


def calibration_error(samples, true, n_levels: int = 11) -> float:
    """Mean absolute deviation between nominal and empirical coverage."""
    nominal, empirical = reliability_curve(samples, true, n_levels)
    return float(np.mean(np.abs(nominal - empirical)))


# --------------------------------------------------------------------------
# front / plume fidelity
# --------------------------------------------------------------------------
def plume_iou(pred, true, threshold: float = 0.1) -> float:
    """Intersection-over-union of the plume masks (saturation > threshold)."""
    pred, true = _np(pred), _np(true)
    pm, tm = pred > threshold, true > threshold
    inter = np.logical_and(pm, tm).sum()
    union = np.logical_or(pm, tm).sum()
    return float(inter / union) if union > 0 else 1.0


def plume_extent_error(pred, true, threshold: float = 0.1) -> float:
    """Relative error in plume area (number of cells above threshold)."""
    pred, true = _np(pred), _np(true)
    a_pred = float((pred > threshold).sum())
    a_true = float((true > threshold).sum())
    return abs(a_pred - a_true) / (a_true + 1e-8)


def summarize(pred, true, samples=None, threshold: float = 0.1) -> dict:
    """Bundle the deterministic (and, if given, probabilistic) metrics."""
    out = {
        "rel_l2": relative_l2(pred, true),
        "rmse": rmse(pred, true),
        "mae": mae(pred, true),
        "plume_iou": plume_iou(pred, true, threshold),
        "plume_extent_err": plume_extent_error(pred, true, threshold),
    }
    if samples is not None:
        out["crps"] = crps_ensemble(samples, true)
        out["calibration_err"] = calibration_error(samples, true)
        out["coverage"] = interval_coverage(samples, true)
    return out
