#!/usr/bin/env python3
"""Train a flow surrogate.

Examples
--------
  python train.py --config configs/default.yaml --model fno --member 0
  python train.py --config configs/default.yaml --model diffusion-unet
  python train.py --config configs/default.yaml --model diffusion-dit --fold 0

FNO is trained by next-step regression with the *pushforward trick*
(Brandstetter et al., ICLR 2022): each batch is a K-step window from a single
simulation, and the model is rolled out K times during training with its own
predictions fed back as input (with ``.detach()`` so gradients flow only
through the current step). A curriculum starts at K=1 for a few warmup epochs
to put the model in a reasonable basin, then switches to K=k_steps_max.

The saturation loss is a **front-aware Huber**: smooth-L1 over the field with
extra weight on cells inside (or near, via dilation) the ground-truth plume,
since the plume is sparse and unweighted MSE is dominated by the zero
background. Pressure stays on plain MSE (smooth, globally coupled, already
accurate). See NOTES.md for the full design rationale.

Diffusion models are trained by epsilon-prediction on single-step transitions
(K=1). The pushforward / front-aware adaptations for those models are
described in NOTES.md but are not yet implemented in this file.
"""
from __future__ import annotations
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neuralflow.config import load_config, ensure_dirs
from neuralflow.data import FlowDataset, FlowSequenceDataset, Normalizer, load_meta
from neuralflow.models import MODEL_KIND, build_model
from neuralflow.utils import EMA, count_params, get_device, save_checkpoint, set_seed


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological dilation of a binary mask via max-pool. radius==0 is identity."""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def _front_aware_huber(pred_sat, target_sat, sat_thr_norm, alpha, dilate, beta):
    """Smooth-L1 with extra weight inside the (dilated) plume region.

    The plume occupies a small fraction of the field; unweighted MSE/L1 is
    dominated by the zero background and lets the model satisfy the loss by
    predicting "no plume". Up-weighting the plume cells (and a small leading-
    edge halo via mask dilation) makes the gradient share comparable.
    """
    mask = (target_sat > sat_thr_norm).float()
    mask = _dilate_mask(mask, dilate)
    elem = F.smooth_l1_loss(pred_sat, target_sat, beta=beta, reduction="none")
    return ((1.0 + alpha * mask) * elem).mean()


def _plume_iou(pred_sat_n, target_sat_n, thr_norm):
    """IoU of the plume mask in normalized saturation space. Returns a scalar tensor."""
    p = (pred_sat_n > thr_norm)
    t = (target_sat_n > thr_norm)
    inter = (p & t).float().sum()
    union = (p | t).float().sum()
    return inter / union.clamp(min=1.0)


def _plot_training_curves(csv_path: str, png_path: str, warmup_k1: int):
    """Re-render the training curve from the CSV log. Cheap; called every epoch."""
    try:
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    except (IOError, ValueError):
        return
    if data.size == 0:
        return
    if data.ndim == 1:
        data = data.reshape(1, -1)
    epoch = data[:, 0]
    train_loss = data[:, 2]
    val_loss = data[:, 5]
    val_iou = data[:, 8] if data.shape[1] > 8 else None

    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.plot(epoch, train_loss, label="train loss", color="C0")
    ax1.plot(epoch, val_loss, label="val loss", color="C1")
    if warmup_k1 > 0:
        ax1.axvline(warmup_k1 + 0.5, color="gray", linestyle="--", linewidth=1,
                    label=f"K=1 -> K>=2 (ep {warmup_k1 + 1})")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")

    if val_iou is not None and np.isfinite(val_iou).any():
        ax2 = ax1.twinx()
        ax2.set_ylabel("val plume IoU")
        ax2.set_ylim(0, 1)
        ax2.plot(epoch, val_iou, label="val IoU", color="C2", alpha=0.7)
        ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True, choices=list(MODEL_KIND))
    ap.add_argument("--member", type=int, default=0, help="FNO deep-ensemble member / seed offset")
    ap.add_argument("--fold", type=int, default=None, help="train on k-fold split <fold>")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    kind = MODEL_KIND[args.model]
    device = get_device()
    set_seed(cfg.data["seed"] + args.member)

    h5 = os.path.join(cfg.paths["data_dir"], "flow.h5")
    meta = load_meta(os.path.join(cfg.paths["data_dir"], "meta.json"))
    split = meta["kfold"][args.fold] if args.fold is not None else meta["splits"]
    norm = Normalizer.from_dict(meta["stats"])
    n_t = cfg.grid["n_timesteps"]
    tc = cfg.train
    amp = bool(tc["amp"]) and device.type == "cuda"

    # ---------- loss / unroll hyperparameters --------------------------------
    sat_w = float(tc.get("sat_loss_weight", 10.0))
    pres_w = float(tc.get("pres_loss_weight", 1.0))
    k_max = int(tc.get("k_steps_max", 1))
    warmup_k1 = int(tc.get("warmup_k1_epochs", 0))
    sat_thr_phys = float(tc.get("sat_threshold", 0.01))
    mask_alpha = float(tc.get("mask_alpha", 20.0))
    mask_dilate = int(tc.get("mask_dilate", 3))
    huber_beta = float(tc.get("huber_beta", 0.05))
    input_noise_sat = float(tc.get("input_noise_sat", 0.0))
    input_noise_pres = float(tc.get("input_noise_pres", 0.0))
    iou_thr_phys = float(cfg.eval.get("front_threshold", 0.1))
    sat_thr_norm = float(norm.norm_sat(sat_thr_phys))
    iou_thr_norm = float(norm.norm_sat(iou_thr_phys))

    # ---------- datasets -----------------------------------------------------
    if kind == "deterministic":
        # K-step sequences (always K=k_max in storage; curriculum picks the unroll length per epoch)
        train_ds = FlowSequenceDataset(h5, split["train"], norm, n_t, k_steps=k_max)
        val_ds = FlowSequenceDataset(h5, split["val"], norm, n_t, k_steps=k_max)
    else:
        # Diffusion models: single-step transitions (K=1).
        train_ds = FlowDataset(h5, split["train"], norm, n_t)
        val_ds = FlowDataset(h5, split["val"], norm, n_t)
    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                          num_workers=tc["num_workers"], drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                        num_workers=tc["num_workers"], pin_memory=True)
    print(f"transitions: train={len(train_ds)}  val={len(val_ds)}")

    # ---------- model / optim ------------------------------------------------
    model = build_model(args.model, cfg).to(device)
    print(f"{args.model}: {count_params(model) / 1e6:.2f}M parameters")

    epochs = args.epochs or tc["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, tc["ema_decay"]) if kind == "generative" else None

    def _k_for_epoch(ep: int) -> int:
        """Curriculum: K=1 warmup for stability, then K=k_max for pushforward."""
        return 1 if ep <= warmup_k1 else k_max

    def _perturb_state(cond):
        """Add per-channel Gaussian noise to the state portion of cond.

        cond layout is [perm, sat_t, pres_t]; perm is static and never noised.
        Active only during training (no-op when model.eval()). This is a
        generalization of the pushforward trick: the model learns to recover
        the next true state from a slightly corrupted input, which directly
        targets robustness to its own autoregressive rollout errors.
        """
        if not model.training:
            return cond
        if input_noise_sat <= 0 and input_noise_pres <= 0:
            return cond
        noise = torch.zeros_like(cond)
        if input_noise_sat > 0:
            noise[:, 1:2] = torch.randn_like(cond[:, 1:2]) * input_noise_sat
        if input_noise_pres > 0:
            noise[:, 2:3] = torch.randn_like(cond[:, 2:3]) * input_noise_pres
        return cond + noise

    def _fno_step(batch, k_unroll, want_iou=False):
        """K-step unrolled loss for the FNO. Returns (total_loss, sat_l, pres_l, iou_or_nan)."""
        cond = batch["cond"].to(device, non_blocking=True)            # (B, 3, H, W)
        targets = batch["targets"].to(device, non_blocking=True)      # (B, K, 2, H, W)
        perm = batch["perm"].to(device, non_blocking=True)            # (B, H, W)
        perm_ch = perm.unsqueeze(1)                                   # (B, 1, H, W)

        sat_losses, pres_losses = [], []
        iou_last = None
        with torch.autocast(device_type=device.type, enabled=amp):
            for step_idx in range(k_unroll):
                pred = model(_perturb_state(cond))                    # (B, 2, H, W)
                tgt = targets[:, step_idx]
                sat_losses.append(
                    _front_aware_huber(pred[:, 0:1], tgt[:, 0:1],
                                       sat_thr_norm, mask_alpha, mask_dilate, huber_beta))
                pres_losses.append(F.mse_loss(pred[:, 1:2], tgt[:, 1:2]))
                if want_iou and step_idx == k_unroll - 1:
                    iou_last = _plume_iou(pred[:, 0:1].detach(),
                                          tgt[:, 0:1].detach(), iou_thr_norm)
                # Pushforward: feed prediction as next conditioning, detach so the
                # gradient flows only through the current step's network call.
                if step_idx < k_unroll - 1:
                    cond = torch.cat([perm_ch, pred.detach()], dim=1)
            sat_l = torch.stack(sat_losses).mean()
            pres_l = torch.stack(pres_losses).mean()
            total = sat_w * sat_l + pres_w * pres_l
        return total, sat_l.detach(), pres_l.detach(), iou_last

    def _diffusion_step(batch):
        cond = batch["cond"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            return model.loss(target, cond)

    tag = args.model
    if args.model == "fno":
        tag += f"_m{args.member}"
    if args.fold is not None:
        tag += f"_fold{args.fold}"
    ckpt = os.path.join(cfg.paths["ckpt_dir"], tag + ".pt")
    csv_path = os.path.join(cfg.paths["results_dir"], f"train_{tag}.csv")
    png_path = os.path.join(cfg.paths["results_dir"], f"train_{tag}.png")
    csv_fh = open(csv_path, "w", newline="", buffering=1)  # line-buffered for tail -f
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["epoch", "k", "train_loss", "train_sat", "train_pres",
                         "val_loss", "val_sat", "val_pres", "val_iou"])

    best = float("inf")
    for ep in range(1, epochs + 1):
        k_now = _k_for_epoch(ep) if kind == "deterministic" else 1
        # K=2 val loss averages step-1 and the (structurally harder) step-2
        # loss, so it is *not* directly comparable to K=1 val loss. Reset the
        # "best" tracker at the curriculum transition so the saved checkpoint
        # is guaranteed to be the best K=2-trained model (rather than an
        # earlier K=1 checkpoint that the K=2 phase can never beat on this
        # metric). Until the transition fires we still save the best K=1 model
        # as a safety net in case the job dies in warmup.
        if kind == "deterministic" and ep == warmup_k1 + 1:
            best = float("inf")
        model.train()
        tr_loss = tr_sat = tr_pres = 0.0
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            if kind == "deterministic":
                loss, sat_l, pres_l, _ = _fno_step(batch, k_now, want_iou=False)
                tr_sat += float(sat_l)
                tr_pres += float(pres_l)
            else:
                loss = _diffusion_step(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            tr_loss += loss.item()
        sched.step()

        model.eval()
        val_loss = val_sat = val_pres = val_iou = 0.0
        n_iou = 0
        with torch.no_grad():
            for batch in val_dl:
                if kind == "deterministic":
                    loss, sat_l, pres_l, iou = _fno_step(batch, k_now, want_iou=True)
                    val_sat += float(sat_l)
                    val_pres += float(pres_l)
                    if iou is not None:
                        val_iou += float(iou)
                        n_iou += 1
                else:
                    loss = _diffusion_step(batch)
                val_loss += loss.item()

        n_tr = max(len(train_dl), 1)
        n_va = max(len(val_dl), 1)
        tr_loss /= n_tr
        val_loss /= n_va
        if kind == "deterministic":
            tr_sat /= n_tr; tr_pres /= n_tr
            val_sat /= n_va; val_pres /= n_va
            val_iou = (val_iou / n_iou) if n_iou else float("nan")
        else:
            tr_sat = tr_pres = val_sat = val_pres = val_iou = float("nan")

        # Per-epoch CSV row + regenerated PNG, so `tail -f` and a scp of the
        # PNG always show the latest curve without waiting for the print interval.
        csv_writer.writerow([ep, k_now, tr_loss, tr_sat, tr_pres,
                             val_loss, val_sat, val_pres, val_iou])
        csv_fh.flush()
        _plot_training_curves(csv_path, png_path, warmup_k1 if kind == "deterministic" else 0)

        if ep == 1 or ep % 10 == 0:
            if kind == "deterministic":
                print(f"epoch {ep:4d}/{epochs}  k={k_now}  "
                      f"train {tr_loss:.5f} (sat {tr_sat:.5f} pres {tr_pres:.5f})  "
                      f"val {val_loss:.5f} (sat {val_sat:.5f} pres {val_pres:.5f})  "
                      f"plume_IoU {val_iou:.3f}", flush=True)
            else:
                print(f"epoch {ep:4d}/{epochs}  train {tr_loss:.5f}  val {val_loss:.5f}",
                      flush=True)

        if val_loss < best:
            best = val_loss
            save_checkpoint(ckpt, ema.shadow if ema else model, opt, ep,
                            {"val_loss": val_loss, "model": args.model, "k_now": k_now})

    csv_fh.close()
    print(f"best val {best:.5f}  ->  {ckpt}", flush=True)
    print(f"training curves: {csv_path}  {png_path}", flush=True)


if __name__ == "__main__":
    main()
