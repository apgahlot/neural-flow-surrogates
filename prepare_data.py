#!/usr/bin/env python3
"""Build simulation-level splits and permeability normalization stats.

Run after `julia export_data.jl` has produced $data_dir/flow.h5. Writes
$data_dir/meta.json (train/val/test splits, k-fold splits, perm stats).
"""
from __future__ import annotations
import argparse
import os

import h5py

from neuralflow.config import load_config, ensure_dirs
from neuralflow.data import compute_perm_stats, compute_pres_stats, kfold_sim_splits, make_sim_splits, save_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    h5_path = os.path.join(cfg.paths["data_dir"], "flow.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"{h5_path} not found - run `julia export_data.jl` first")

    with h5py.File(h5_path, "r") as f:
        sat_shape = f["saturation"].shape
        n_sims = sat_shape[0]
        perm_all_zero = float(f["permeability"][:].std()) == 0.0
    print(f"flow.h5 saturation shape (N,T,H,W) = {sat_shape}")
    if perm_all_zero:
        print("WARNING: permeability is all zeros - re-run export_data.jl with "
              "correct raw.perm_var / raw.idx_var (see `--inspect`).")

    data = cfg.data
    splits = make_sim_splits(n_sims, data["split"], data["seed"])
    kfold = kfold_sim_splits(n_sims, data["kfold"], data["seed"])
    stats = compute_perm_stats(h5_path, splits["train"], data["perm_log_transform"])
    stats.update(compute_pres_stats(h5_path, splits["train"]))
    stats["sat_max"] = float(data["saturation_clip"][1])

    meta_path = os.path.join(cfg.paths["data_dir"], "meta.json")
    save_meta(meta_path, splits, stats, kfold)

    print("splits (by simulation):", {k: len(v) for k, v in splits.items()})
    print(f"k-fold: {len(kfold)} folds")
    print("perm stats:", stats)
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
