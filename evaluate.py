#!/usr/bin/env python3
"""Evaluate a trained joint-state surrogate by autoregressive rollout over the test set.

Reports per-channel (saturation, pressure) relative L2 across the rollout,
plume IoU (saturation), inference speed, and CRPS / interval coverage /
calibration error for FNO ensembles and generative models.

Examples
--------
  python evaluate.py --config configs/default.yaml --model fno
  python evaluate.py --config configs/default.yaml --model diffusion-unet --samples 32
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from neuralflow import metrics
from neuralflow.config import load_config
from neuralflow.data import Normalizer, load_meta, load_trajectory
from neuralflow.inference import rollout_diffusion, rollout_fno
from neuralflow.models import MODEL_KIND, build_model
from neuralflow.utils import Timer, get_device, load_checkpoint


def load_trained(name, cfg, ckpt, device):
    model = build_model(name, cfg)
    load_checkpoint(ckpt, model, map_location=device)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True, choices=list(MODEL_KIND))
    ap.add_argument("--samples", type=int, default=None, help="generative UQ samples")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    h5 = os.path.join(cfg.paths["data_dir"], "flow.h5")
    meta = load_meta(os.path.join(cfg.paths["data_dir"], "meta.json"))
    norm = Normalizer.from_dict(meta["stats"])
    test_sims = meta["splits"]["test"]
    n_steps = cfg.eval["rollout_steps"]
    thr = cfg.eval["front_threshold"]
    ckdir = cfg.paths["ckpt_dir"]

    if args.model == "fno":
        paths = sorted(glob.glob(os.path.join(ckdir, "fno_m*.pt"))) or [os.path.join(ckdir, "fno.pt")]
        members = [load_trained("fno", cfg, p, device) for p in paths]
        print(f"FNO deep ensemble: {len(members)} member(s)")
    else:
        diff = load_trained(args.model, cfg, os.path.join(ckdir, f"{args.model}.pt"), device)
        n_samples = args.samples or cfg.eval["uq_samples"]

    rl2_sat = np.zeros(n_steps + 1)
    rl2_pres = np.zeros(n_steps + 1)
    iou = np.zeros(n_steps + 1)
    crps_sat, crps_pres = [], []
    cal_sat, cal_pres = [], []
    cov_sat, cov_pres = [], []
    times = []

    for sim in test_sims:
        perm_np, sat_raw, pres_raw = load_trajectory(h5, sim, norm)
        perm = torch.from_numpy(perm_np)
        state0 = torch.from_numpy(np.stack(
            [norm.norm_sat(sat_raw[0]), norm.norm_pres(pres_raw[0])], axis=0))
        true_sat = sat_raw[: n_steps + 1]
        true_pres = pres_raw[: n_steps + 1]

        with Timer() as tm:
            if args.model == "fno":
                traj = rollout_fno(members, perm, state0, n_steps, device)
            else:
                traj = rollout_diffusion(diff, perm, state0, n_steps, n_samples, device,
                                         cfg.diffusion["sampler"], cfg.diffusion["sampler_steps"])
        times.append(tm.seconds)

        sat_traj = norm.denorm_sat(traj[:, :, 0, :, :])         # (M, n_steps+1, H, W) physical
        pres_traj = norm.denorm_pres(traj[:, :, 1, :, :])
        sat_mean = sat_traj.mean(axis=0)
        pres_mean = pres_traj.mean(axis=0)

        rl2_sat += np.array([metrics.relative_l2(sat_mean[k][None], true_sat[k][None]) for k in range(n_steps + 1)])
        rl2_pres += np.array([metrics.relative_l2(pres_mean[k][None], true_pres[k][None]) for k in range(n_steps + 1)])
        iou += np.array([metrics.plume_iou(sat_mean[k], true_sat[k], thr) for k in range(n_steps + 1)])

        if sat_traj.shape[0] > 1:
            crps_sat.append(metrics.crps_ensemble(sat_traj, true_sat))
            crps_pres.append(metrics.crps_ensemble(pres_traj, true_pres))
            cal_sat.append(metrics.calibration_error(sat_traj, true_sat))
            cal_pres.append(metrics.calibration_error(pres_traj, true_pres))
            cov_sat.append(metrics.interval_coverage(sat_traj, true_sat))
            cov_pres.append(metrics.interval_coverage(pres_traj, true_pres))

    n = max(len(test_sims), 1)
    rl2_sat /= n
    rl2_pres /= n
    iou /= n
    results = {
        "model": args.model,
        "n_test_sims": len(test_sims),
        "rollout_steps": n_steps,
        "saturation": {
            "rel_l2_per_step": rl2_sat.round(5).tolist(),
            "rel_l2_final": float(rl2_sat[-1]),
            "rel_l2_mean": float(rl2_sat.mean()),
            "plume_iou_per_step": iou.round(4).tolist(),
            "plume_iou_mean": float(iou.mean()),
        },
        "pressure": {
            "rel_l2_per_step": rl2_pres.round(5).tolist(),
            "rel_l2_final": float(rl2_pres[-1]),
            "rel_l2_mean": float(rl2_pres.mean()),
        },
        "rollout_seconds_mean": float(np.mean(times)),
    }
    jutul = cfg.eval.get("jutul_reference_seconds")
    if jutul:
        results["speedup_vs_jutuldarcy"] = float(jutul) / float(np.mean(times))
    if crps_sat:
        results["saturation"].update({
            "crps_mean": float(np.mean(crps_sat)),
            "calibration_error_mean": float(np.mean(cal_sat)),
            "coverage": {str(k): float(np.mean([c[k] for c in cov_sat])) for k in sorted(cov_sat[0])},
        })
        results["pressure"].update({
            "crps_mean": float(np.mean(crps_pres)),
            "calibration_error_mean": float(np.mean(cal_pres)),
            "coverage": {str(k): float(np.mean([c[k] for c in cov_pres])) for k in sorted(cov_pres[0])},
        })

    os.makedirs(cfg.paths["results_dir"], exist_ok=True)
    out = os.path.join(cfg.paths["results_dir"], f"eval_{args.model}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)

    steps = np.arange(n_steps + 1)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    ax[0].plot(steps, rl2_sat, "o-")
    ax[0].set(xlabel="rollout step", ylabel="relative L2", title="saturation error")
    ax[1].plot(steps, rl2_pres, "o-", color="darkorange")
    ax[1].set(xlabel="rollout step", ylabel="relative L2", title="pressure error")
    ax[2].plot(steps, iou, "s-", color="seagreen")
    ax[2].set(xlabel="rollout step", ylabel="plume IoU", title="front fidelity (sat)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{args.model} - autoregressive rollout ({len(test_sims)} test sims)")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.paths["results_dir"], f"eval_{args.model}_rollout.png"), dpi=130)

    print(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
