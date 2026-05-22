#!/usr/bin/env python3
"""Detailed autoregressive rollout for a single case, with field-level figures
for both saturation and pressure.

Default case is the held-out ground-truth permeability. Produces one panel per
channel (ground truth / prediction / absolute error / uncertainty) at four
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


def panel(true, pred_mean, pred_std, show_steps, title, cmaps, out_path):
    rows = ["ground truth", "prediction", "abs error", "uncertainty (std)"]
    fig, ax = plt.subplots(4, len(show_steps), figsize=(3.1 * len(show_steps), 9.6), squeeze=False)
    for j, k in enumerate(show_steps):
        fields = [true[k], pred_mean[k], np.abs(pred_mean[k] - true[k]), pred_std[k]]
        for r, (fld, cm) in enumerate(zip(fields, cmaps)):
            im = ax[r][j].imshow(fld, cmap=cm, aspect="auto")
            ax[r][j].set_xticks([])
            ax[r][j].set_yticks([])
            if j == 0:
                ax[r][j].set_ylabel(rows[r], fontsize=9)
            if r == 0:
                ax[r][j].set_title(f"t = {k}", fontsize=9)
            fig.colorbar(im, ax=ax[r][j], fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


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
        perm_np, sat_raw, pres_raw = load_trajectory(h5, 0, norm, group="gt")
        case = "gt"
    else:
        perm_np, sat_raw, pres_raw = load_trajectory(h5, int(args.case), norm)
        case = f"sim{args.case}"
    perm = torch.from_numpy(perm_np)
    state0 = torch.from_numpy(np.stack(
        [norm.norm_sat(sat_raw[0]), norm.norm_pres(pres_raw[0])], axis=0))
    true_sat = sat_raw[: n_steps + 1]
    true_pres = pres_raw[: n_steps + 1]

    if args.model == "fno":
        paths = sorted(glob.glob(os.path.join(ckdir, "fno_m*.pt"))) or [os.path.join(ckdir, "fno.pt")]
        members = [load_trained("fno", cfg, p, device) for p in paths]
        traj = rollout_fno(members, perm, state0, n_steps, device)
    else:
        diff = load_trained(args.model, cfg, os.path.join(ckdir, f"{args.model}.pt"), device)
        n_samples = args.samples or cfg.eval["uq_samples"]
        traj = rollout_diffusion(diff, perm, state0, n_steps, n_samples, device,
                                 cfg.diffusion["sampler"], cfg.diffusion["sampler_steps"])

    sat_traj = norm.denorm_sat(traj[:, :, 0, :, :])
    pres_traj = norm.denorm_pres(traj[:, :, 1, :, :])
    sat_mean, sat_std = ensemble_stats(sat_traj)
    pres_mean, pres_std = ensemble_stats(pres_traj)

    show = sorted(set(int(s) for s in np.linspace(1, n_steps, 4)))
    os.makedirs(cfg.paths["results_dir"], exist_ok=True)

    panel(true_sat, sat_mean, sat_std, show,
          f"{args.model} - saturation rollout ({case})",
          ["viridis", "viridis", "magma", "cividis"],
          os.path.join(cfg.paths["results_dir"], f"rollout_{args.model}_{case}_saturation.png"))
    panel(true_pres, pres_mean, pres_std, show,
          f"{args.model} - pressure rollout ({case})",
          ["plasma", "plasma", "magma", "cividis"],
          os.path.join(cfg.paths["results_dir"], f"rollout_{args.model}_{case}_pressure.png"))

    rl2 = lambda pred, true: np.mean([
        np.linalg.norm(pred[k] - true[k]) / (np.linalg.norm(true[k]) + 1e-8) for k in range(n_steps + 1)])
    print(f"mean rollout rel-L2  saturation: {rl2(sat_mean, true_sat):.4f}   "
          f"pressure: {rl2(pres_mean, true_pres):.4f}")
    print("wrote two figures to", cfg.paths["results_dir"])


if __name__ == "__main__":
    main()
