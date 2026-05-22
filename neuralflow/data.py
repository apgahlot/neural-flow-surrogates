"""Datasets and normalization for autoregressive flow surrogates.

The surrogate is autoregressive: given a permeability field K and the saturation
state S_t, predict S_{t+1}. Each of the 128 simulations therefore yields
(n_timesteps - 1) training transitions. Splits are made at the *simulation*
level so no trajectory leaks between train / val / test.
"""
from __future__ import annotations
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def make_sim_splits(n_sims: int, fracs: dict, seed: int) -> dict:
    """Random split of simulation indices into train / val / test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_sims)
    n_tr = int(round(fracs["train"] * n_sims))
    n_va = int(round(fracs["val"] * n_sims))
    return {
        "train": sorted(int(i) for i in idx[:n_tr]),
        "val": sorted(int(i) for i in idx[n_tr:n_tr + n_va]),
        "test": sorted(int(i) for i in idx[n_tr + n_va:]),
    }


def kfold_sim_splits(n_sims: int, k: int, seed: int) -> list[dict]:
    """k-fold cross-validation splits over simulation indices (small-data regime)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_sims)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        test = sorted(int(j) for j in folds[i])
        train = sorted(int(j) for f in range(k) if f != i for j in folds[f])
        splits.append({"train": train, "val": test, "test": test})
    return splits


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------
class Normalizer:
    """perm : log10 + standardize.  sat : [0,1] -> [-1,1].  pres : min-max -> [-1,1]."""

    def __init__(self, perm_mean, perm_std, perm_log=True, pres_min=0.0, pres_max=1.0):
        self.perm_mean = float(perm_mean)
        self.perm_std = float(perm_std) if perm_std else 1.0
        self.perm_log = bool(perm_log)
        self.pres_min = float(pres_min)
        self.pres_max = float(pres_max)
        self._pres_range = max(self.pres_max - self.pres_min, 1e-12)

    def norm_perm(self, x):
        if self.perm_log:
            x = np.log10(np.clip(x, 1e-12, None))
        return (x - self.perm_mean) / self.perm_std

    def norm_sat(self, x):
        return x * 2.0 - 1.0

    def denorm_sat(self, x):
        return (x + 1.0) * 0.5

    def norm_pres(self, x):
        return 2.0 * (x - self.pres_min) / self._pres_range - 1.0

    def denorm_pres(self, x):
        return (x + 1.0) * 0.5 * self._pres_range + self.pres_min

    def to_dict(self) -> dict:
        return {"perm_mean": self.perm_mean, "perm_std": self.perm_std, "perm_log": self.perm_log,
                "pres_min": self.pres_min, "pres_max": self.pres_max}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(d["perm_mean"], d["perm_std"], d["perm_log"],
                   d.get("pres_min", 0.0), d.get("pres_max", 1.0))


def compute_perm_stats(h5_path: str, train_sims: list[int], perm_log: bool) -> dict:
    """Mean/std of (log-)permeability over the training simulations only."""
    with h5py.File(h5_path, "r") as f:
        perm = f["permeability"][sorted(train_sims)].astype("float64")
    if perm_log:
        perm = np.log10(np.clip(perm, 1e-12, None))
    return {"perm_mean": float(perm.mean()), "perm_std": float(perm.std()), "perm_log": perm_log}


def compute_pres_stats(h5_path: str, train_sims: list[int]) -> dict:
    """min and max of pressure across the training simulations and all timesteps."""
    mn, mx = float("inf"), float("-inf")
    with h5py.File(h5_path, "r") as f:
        for s in sorted(train_sims):
            block = f["pressure"][s][...]
            mn = min(mn, float(block.min()))
            mx = max(mx, float(block.max()))
    return {"pres_min": mn, "pres_max": mx}


# --------------------------------------------------------------------------
# meta (splits + stats) persistence
# --------------------------------------------------------------------------
def save_meta(path: str, splits: dict, stats: dict, kfold: list | None = None) -> None:
    with open(path, "w") as fh:
        json.dump({"splits": splits, "stats": stats, "kfold": kfold or []}, fh, indent=2)


def load_meta(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
class FlowDataset(Dataset):
    """Autoregressive transition pairs over the coupled (saturation, pressure) state.

    __getitem__ returns:
        cond   : (3, H, W)  channels = [permeability, S_t, P_t]
        target : (2, H, W)  channels = [S_{t+1}, P_{t+1}]
    """

    def __init__(self, h5_path: str, sim_indices, normalizer: Normalizer, n_timesteps: int):
        self.h5_path = h5_path
        self.sims = list(sim_indices)
        self.norm = normalizer
        self.T = n_timesteps
        self.pairs = [(s, t) for s in self.sims for t in range(self.T - 1)]
        self._h5 = None  # opened lazily, per-worker

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        sim, t = self.pairs[i]
        f = self._file()
        perm = self.norm.norm_perm(f["permeability"][sim].astype("float32"))
        sat_t = self.norm.norm_sat(f["saturation"][sim, t].astype("float32"))
        sat_t1 = self.norm.norm_sat(f["saturation"][sim, t + 1].astype("float32"))
        pres_t = self.norm.norm_pres(f["pressure"][sim, t].astype("float32"))
        pres_t1 = self.norm.norm_pres(f["pressure"][sim, t + 1].astype("float32"))
        cond = np.stack([perm, sat_t, pres_t], axis=0).astype("float32")
        target = np.stack([sat_t1, pres_t1], axis=0).astype("float32")
        return {
            "cond": torch.from_numpy(cond),
            "target": torch.from_numpy(target),
            "sim": sim,
            "t": t,
        }


def load_trajectory(h5_path: str, sim: int, normalizer: Normalizer, group: str = "saturation"):
    """Return (perm_norm (H,W), sat_raw (T,H,W), pres_raw (T,H,W)) in physical units."""
    with h5py.File(h5_path, "r") as f:
        if group == "gt":
            perm = f["gt_permeability"][:]
            sat = f["gt_saturation"][:]
            pres = f["gt_pressure"][:]
        else:
            perm = f["permeability"][sim]
            sat = f["saturation"][sim]
            pres = f["pressure"][sim]
    return (normalizer.norm_perm(perm.astype("float32")),
            sat.astype("float32"),
            pres.astype("float32"))
