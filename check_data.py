#!/usr/bin/env python3
"""Pre-training sanity check.

Reports flow.h5 contents, splits, normalization stats and roundtrips, one
FlowDataset sample, one DataLoader batch, and a single forward pass through
each model on the current device. Run after `prepare_data.py` and before the
first `train.py` invocation.
"""
from __future__ import annotations
import argparse
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from neuralflow.config import load_config
from neuralflow.data import FlowDataset, Normalizer, load_meta
from neuralflow.models import MODEL_KIND, build_model
from neuralflow.utils import count_params, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    h5_path = os.path.join(cfg.paths["data_dir"], "flow.h5")
    meta_path = os.path.join(cfg.paths["data_dir"], "meta.json")

    # ---- flow.h5 contents -------------------------------------------------
    print("\n--- flow.h5 ---")
    total_bytes = 0
    with h5py.File(h5_path, "r") as f:
        for k in f:
            d = f[k]
            nbytes = int(np.prod(d.shape)) * d.dtype.itemsize
            total_bytes += nbytes
            print(f"  {k:<18} {str(tuple(d.shape)):<28} {str(d.dtype):<8} {nbytes / 1e6:>9.1f} MB")
    print(f"  total: {total_bytes / 1e9:.2f} GB on disk")

    # ---- splits + autoregressive-pair counts -----------------------------
    meta = json.load(open(meta_path))
    T = cfg.grid["n_timesteps"]
    print("\n--- splits (simulation-level)  ->  autoregressive pairs ---")
    for name, sims in meta["splits"].items():
        print(f"  {name:<6} {len(sims):>4} sims   ->  {len(sims) * (T - 1):>5} pairs")
    print(f"  k-fold: {len(meta['kfold'])} folds")

    # ---- normalization stats + roundtrips --------------------------------
    print("\n--- normalization stats ---")
    for k, v in meta["stats"].items():
        print(f"  {k:<12} {v}")
    norm = Normalizer.from_dict(meta["stats"])
    sat_vals = np.array([0.0, 0.45, 0.9], dtype=np.float32)
    pres_vals = np.array([norm.pres_min, 0.5 * (norm.pres_min + norm.pres_max), norm.pres_max],
                         dtype=np.float64)
    sat_rt = norm.denorm_sat(norm.norm_sat(sat_vals))
    pres_rt = norm.denorm_pres(norm.norm_pres(pres_vals))
    print(f"  sat  roundtrip {sat_vals.tolist()}  -> {np.round(sat_rt, 6).tolist()}")
    print(f"  pres roundtrip [min, mid, max]      -> {np.round(pres_rt, 1).tolist()}")

    # ---- FlowDataset sample ----------------------------------------------
    print("\n--- FlowDataset[0] (training split) ---")
    ds = FlowDataset(h5_path, meta["splits"]["train"], norm, T)
    item = ds[0]
    cond, target = item["cond"], item["target"]
    print(f"  cond   {tuple(cond.shape)}  range [{cond.min():.3f}, {cond.max():.3f}]   sim={item['sim']}  t={item['t']}")
    print(f"  target {tuple(target.shape)}  range [{target.min():.3f}, {target.max():.3f}]")
    print(f"  len(dataset) = {len(ds)} pairs")

    # ---- one DataLoader batch --------------------------------------------
    print("\n--- DataLoader batch ---")
    bs = cfg.train["batch_size"]
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    cond, target = batch["cond"], batch["target"]
    mb = (cond.element_size() * cond.nelement() + target.element_size() * target.nelement()) / 1e6
    print(f"  cond   {tuple(cond.shape)}  {cond.dtype}")
    print(f"  target {tuple(target.shape)}  {target.dtype}")
    print(f"  per-batch tensors: {mb:.1f} MB")

    # ---- model forward / loss check on the current device ----------------
    print("\n--- model forward + loss on real batch ---")
    device = get_device()
    print(f"  device: {device}")
    cond_d = cond.to(device)
    target_d = target.to(device)
    for name, kind in MODEL_KIND.items():
        m = None
        try:
            m = build_model(name, cfg).to(device)
            if kind == "deterministic":
                loss = torch.nn.functional.mse_loss(m(cond_d), target_d)
            else:
                loss = m.loss(target_d, cond_d)
            mem = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
            print(f"  {name:<18} {count_params(m) / 1e6:>6.2f}M params   "
                  f"loss = {loss.item():.4f}" + (f"   peak GPU = {mem:.2f} GB" if mem else ""))
        except Exception as e:
            print(f"  {name:<18} ERROR: {type(e).__name__}: {e}")
        if m is not None:
            del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    print()


if __name__ == "__main__":
    main()
