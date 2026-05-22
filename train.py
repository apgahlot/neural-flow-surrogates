#!/usr/bin/env python3
"""Train a flow surrogate.

Examples
--------
  python train.py --config configs/default.yaml --model fno --member 0
  python train.py --config configs/default.yaml --model diffusion-unet
  python train.py --config configs/default.yaml --model diffusion-dit --fold 0

FNO is trained by next-step regression (train one model per `--member` for a
deep ensemble). Diffusion models are trained by epsilon-prediction.
"""
from __future__ import annotations
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neuralflow.config import load_config, ensure_dirs
from neuralflow.data import FlowDataset, Normalizer, load_meta
from neuralflow.models import MODEL_KIND, build_model
from neuralflow.utils import EMA, count_params, get_device, save_checkpoint, set_seed


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

    train_ds = FlowDataset(h5, split["train"], norm, n_t)
    val_ds = FlowDataset(h5, split["val"], norm, n_t)
    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                          num_workers=tc["num_workers"], drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                        num_workers=tc["num_workers"], pin_memory=True)
    print(f"transitions: train={len(train_ds)}  val={len(val_ds)}")

    model = build_model(args.model, cfg).to(device)
    print(f"{args.model}: {count_params(model) / 1e6:.2f}M parameters")

    epochs = args.epochs or tc["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, tc["ema_decay"]) if kind == "generative" else None

    def step(batch):
        cond = batch["cond"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            if kind == "deterministic":
                return F.mse_loss(model(cond), target)
            return model.loss(target, cond)

    tag = args.model
    if args.model == "fno":
        tag += f"_m{args.member}"
    if args.fold is not None:
        tag += f"_fold{args.fold}"
    ckpt = os.path.join(cfg.paths["ckpt_dir"], tag + ".pt")

    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            loss = step(batch)
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
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                val_loss += step(batch).item()
        tr_loss /= max(len(train_dl), 1)
        val_loss /= max(len(val_dl), 1)
        if ep == 1 or ep % 10 == 0:
            print(f"epoch {ep:4d}/{epochs}  train {tr_loss:.5f}  val {val_loss:.5f}")

        if val_loss < best:
            best = val_loss
            save_checkpoint(ckpt, ema.shadow if ema else model, opt, ep,
                            {"val_loss": val_loss, "model": args.model})

    print(f"best val {best:.5f}  ->  {ckpt}")


if __name__ == "__main__":
    main()
