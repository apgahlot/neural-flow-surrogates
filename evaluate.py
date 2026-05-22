#!/usr/bin/env python3
"""Evaluate a trained surrogate by autoregressive rollout over the test set.

Reports per-step relative L2, plume IoU, inference speed, and (for generative
models / FNO ensembles) CRPS, interval coverage and calibration error.
Writes results JSON + an error-growth figure to $results_dir.

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

    rl2 = np.zeros(n_steps + 1)
    iou = np.zeros(n_steps + 1)
    crps_all, cal_all, cov_all, times = [], [], [], []

    for sim in test_sims:
        perm_np, sat_raw = load_trajectory(h5, sim, norm)
        perm = torch.from_numpy(perm_np)
        sat0 = torch.from_numpy(norm.norm_sat(sat_raw[0]))
        true = sat_raw[: n_steps + 1]

        with Timer() as tm:
            if args.model == "fno":
                traj = rollout_fno(members, perm, sat0, n_steps, device)
            else:
                traj = rollout_diffusion(diff, perm, sat0, n_steps, n_samples, device,
                                         cfg.diffusion["sampler"], cfg.diffusion["sampler_steps"])
        times.append(tm.seconds)

        traj = norm.denorm_sat(traj)                       # (M, n_steps+1, H, W) in [0,1]
        mean = traj.mean(axis=0)
        rl2 += np.array([metrics.relative_l2(mean[k][None], true[k][None]) for k in range(n_steps + 1)])
        iou += np.array([metrics.plume_iou(mean[k], true[k], thr) for k in range(n_steps + 1)])
        if traj.shape[0] > 1:
            crps_all.append(metrics.crps_ensemble(traj, true))
            cal_all.append(metrics.calibration_error(traj, true))
            cov_all.append(metrics.interval_coverage(traj, true))

    n = max(len(test_sims), 1)
    rl2 /= n
    iou /= n
    results = {
        "model": args.model,
        "n_test_sims": len(test_sims),
        "rollout_steps": n_steps,
        "rel_l2_per_step": rl2.round(5).tolist(),
        "rel_l2_final": float(rl2[-1]),
        "rel_l2_mean": float(rl2.mean()),
        "plume_iou_mean": float(iou.mean()),
        "rollout_seconds_mean": float(np.mean(times)),
    }
    jutul = cfg.eval.get("jutul_reference_seconds")
    if jutul:
        results["speedup_vs_jutuldarcy"] = float(jutul) / float(np.mean(times))
    if crps_all:
        results["crps_mean"] = float(np.mean(crps_all))
        results["calibration_error_mean"] = float(np.mean(cal_all))
        keys = sorted(cov_all[0].keys())
        results["coverage"] = {str(k): float(np.mean([c[k] for c in cov_all])) for k in keys}

    os.makedirs(cfg.paths["results_dir"], exist_ok=True)
    out = os.path.join(cfg.paths["results_dir"], f"eval_{args.model}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)

    steps = np.arange(n_steps + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(steps, rl2, "o-")
    ax[0].set(xlabel="rollout step", ylabel="relative L2", title="error growth")
    ax[1].plot(steps, iou, "s-", color="seagreen")
    ax[1].set(xlabel="rollout step", ylabel="plume IoU", title="front fidelity")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{args.model} - autoregressive rollout ({len(test_sims)} test sims)")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.paths["results_dir"], f"eval_{args.model}_rollout.png"), dpi=130)

    print(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
