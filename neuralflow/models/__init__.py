"""Model factory."""
from __future__ import annotations

from .fno import FNO2d
from .unet import CondUNet
from .dit import DiT
from .diffusion import GaussianDiffusion

# "deterministic" models are trained by regression; "generative" by diffusion.
MODEL_KIND = {
    "fno": "deterministic",
    "diffusion-unet": "generative",
    "diffusion-dit": "generative",
}


def build_model(name: str, cfg):
    """Construct a model from its name and the experiment config."""
    grid, mc, dc = cfg["grid"], cfg["model"], cfg["diffusion"]

    # joint (saturation, pressure) state -> 2 target channels; cond = [perm, S_t, P_t] -> 3 channels
    if name == "fno":
        m = mc["fno"]
        # residual_per_channel (list) takes precedence over scalar residual
        if "residual_per_channel" in m:
            residual = list(m["residual_per_channel"])
        else:
            residual = bool(m.get("residual", True))
        return FNO2d(in_ch=3, out_ch=2, width=m["width"],
                     modes=tuple(m["modes"]), n_layers=m["n_layers"],
                     residual=residual)

    if name == "diffusion-unet":
        u = mc["unet"]
        denoiser = CondUNet(target_ch=2, cond_ch=3, base=u["base_channels"],
                            channel_mults=tuple(u["channel_mults"]))
        return GaussianDiffusion(denoiser, timesteps=dc["timesteps"], schedule=dc["schedule"])

    if name == "diffusion-dit":
        d = mc["dit"]
        denoiser = DiT(height=grid["height"], width=grid["width"], target_ch=2, cond_ch=3,
                       patch_size=d["patch_size"], hidden=d["hidden_size"],
                       depth=d["depth"], heads=d["num_heads"])
        return GaussianDiffusion(denoiser, timesteps=dc["timesteps"], schedule=dc["schedule"])

    raise ValueError(f"unknown model '{name}' (choose from {list(MODEL_KIND)})")


__all__ = ["FNO2d", "CondUNet", "DiT", "GaussianDiffusion", "build_model", "MODEL_KIND"]
