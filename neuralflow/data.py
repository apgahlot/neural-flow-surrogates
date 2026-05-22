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
    """Permeability: optional log10 then standardize. Saturation: [0,1] -> [-1,1]."""

    def __init__(self, perm_mean: float, perm_std: float, perm_log: bool = True):
        self.perm_mean = float(perm_mean)
        self.perm_std = float(perm_std) if perm_std else 1.0
        self.perm_log = perm_log

    def norm_perm(self, x):
        if self.perm_log:
            x = np.log10(np.clip(x, 1e-12, None))
        return (x - self.perm_mean) / self.perm_std

    def norm_sat(self, x):
        return x * 2.0 - 1.0

    def denorm_sat(self, x):
        return (x + 1.0) * 0.5

    def to_dict(self) -> dict:
        return {"perm_mean": self.perm_mean, "perm_std": self.perm_std, "perm_log": self.perm_log}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(d["perm_mean"], d["perm_std"], d["perm_log"])


def compute_perm_stats(h5_path: str, train_sims: list[int], perm_log: bool) -> dict:
    """Mean/std of (log-)permeability over the training simulations only."""
    with h5py.File(h5_path, "r") as f:
        perm = f["permeability"][sorted(train_sims)].astype("float64")
    if perm_log:
        perm = np.log10(np.clip(perm, 1e-12, None))
    return {"perm_mean": float(perm.mean()), "perm_std": float(perm.std()), "perm_log": perm_log}


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
    """Autoregressive transition pairs.

    __getitem__ returns:
        cond   : (2, H, W)  channels = [permeability, S_t]
        target : (1, H, W)  = S_{t+1}
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
        cond = np.stack([perm, sat_t], axis=0).astype("float32")
        target = sat_t1[None].astype("float32")
        return {
            "cond": torch.from_numpy(cond),
            "target": torch.from_numpy(target),
            "sim": sim,
            "t": t,
        }


def load_trajectory(h5_path: str, sim: int, normalizer: Normalizer, group: str = "saturation"):
    """Return (normalized permeability (H,W), raw saturation trajectory (T,H,W))."""
    with h5py.File(h5_path, "r") as f:
        if group == "gt":
            perm = f["gt_permeability"][:]
            sat = f["gt_saturation"][:]
        else:
            perm = f["permeability"][sim]
            sat = f["saturation"][sim]
    return normalizer.norm_perm(perm.astype("float32")), sat.astype("float32")
