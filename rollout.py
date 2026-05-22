#!/usr/bin/env python3
"""Detailed autoregressive rollout for a single case, with field-level figures.

Default case is the held-out ground-truth permeability. Produces a panel of
ground truth / prediction / absolute error / predictive uncertainty at four
rollout times.

Examples
--------
  python rollout.py --config configs/default.yaml --model diffusion-unet --samples 32
  python rollout.py --config configs/default.yaml --model fno --case 7
"""
from __future__ import annotations
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from neuralflow.config import load_config
from neuralflow.data import Normalizer, load_meta, load_trajectory
from neuralflow.inference import ensemble_stats, rollout_diffusion, rollout_fno
from neuralflow.models import MODEL_KIND, build_model
from neuralflow.utils import get_device, load_checkpoint


def load_trained(name, cfg, ckpt, device):
    model = build_model(name, cfg)
    load_checkpoint(ckpt, model, map_location=device)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True, choices=list(MODEL_KIND))
    ap.add_argument("--case", default="gt", help="'gt' or a simulation index")
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    h5 = os.path.join(cfg.paths["data_dir"], "flow.h5")
    meta = load_meta(os.path.join(cfg.paths["data_dir"], "meta.json"))
    norm = Normalizer.from_dict(meta["stats"])
    n_steps = cfg.eval["rollout_steps"]
    ckdir = cfg.paths["ckpt_dir"]

    if args.case == "gt":
        perm_np, sat_raw = load_trajectory(h5, 0, norm, group="gt")
        case = "gt"
    else:
        perm_np, sat_raw = load_trajectory(h5, int(args.case), norm)
        case = f"sim{args.case}"
    perm = torch.from_numpy(perm_np)
    sat0 = torch.from_numpy(norm.norm_sat(sat_raw[0]))
    true = sat_raw[: n_steps + 1]

    if args.model == "fno":
        paths = sorted(glob.glob(os.path.join(ckdir, "fno_m*.pt"))) or [os.path.join(ckdir, "fno.pt")]
        members = [load_trained("fno", cfg, p, device) for p in paths]
        traj = rollout_fno(members, perm, sat0, n_steps, device)
    else:
        diff = load_trained(args.model, cfg, os.path.join(ckdir, f"{args.model}.pt"), device)
        n_samples = args.samples or cfg.eval["uq_samples"]
        traj = rollout_diffusion(diff, perm, sat0, n_steps, n_samples, device,
                                 cfg.diffusion["sampler"], cfg.diffusion["sampler_steps"])

    traj = norm.denorm_sat(traj)
    mean, std = ensemble_stats(traj)

    show = sorted(set(int(s) for s in np.linspace(1, n_steps, 4)))
    rows = ["ground truth", "prediction", "abs error", "uncertainty (std)"]
    cmaps = ["viridis", "viridis", "magma", "cividis"]
    fig, ax = plt.subplots(4, len(show), figsize=(3.1 * len(show), 9.6), squeeze=False)
    for j, k in enumerate(show):
        fields = [true[k], mean[k], np.abs(mean[k] - true[k]), std[k]]
        for r, (fld, cm) in enumerate(zip(fields, cmaps)):
            im = ax[r][j].imshow(fld, cmap=cm, aspect="auto")
            ax[r][j].set_xticks([])
            ax[r][j].set_yticks([])
            if j == 0:
                ax[r][j].set_ylabel(rows[r], fontsize=9)
            if r == 0:
                ax[r][j].set_title(f"t = {k}", fontsize=9)
            fig.colorbar(im, ax=ax[r][j], fraction=0.046, pad=0.04)
    fig.suptitle(f"{args.model} - autoregressive rollout ({case})")
    fig.tight_layout()

    os.makedirs(cfg.paths["results_dir"], exist_ok=True)
    out = os.path.join(cfg.paths["results_dir"], f"rollout_{args.model}_{case}.png")
    fig.savefig(out, dpi=130)
    print(f"mean rel-L2 over rollout: "
          f"{np.mean([np.linalg.norm(mean[k] - true[k]) / (np.linalg.norm(true[k]) + 1e-8) for k in range(n_steps + 1)]):.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
