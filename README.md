# neural-flow-surrogates

**Deterministic vs. generative neural surrogates for two-phase flow in porous media — a data-efficient benchmark with uncertainty quantification.**

Two-phase (CO₂ / brine) flow in porous media is governed by a Darcy-flow PDE system and is
expensive to simulate. This project trains neural surrogates that replace the numerical
simulator and benchmarks them on three axes that actually matter in practice — **accuracy,
inference speed, and uncertainty calibration** — in the **small-data regime** (≈10²
simulations), which is the regime most engineering teams are actually in.

The surrogate is **autoregressive over the coupled (saturation, pressure) state** —
because JutulDarcy solves a coupled multiphase-flow system, learning *S* without *P*
throws away the variable that actually drives the flow. Given a permeability field `K`
and the current state `(Sₜ, Pₜ)`, the model predicts `(Sₜ₊₁, Pₜ₊₁)`, then rolls out
the full horizon. Treating each simulation as 23 one-step transitions turns a handful
of trajectories into thousands of training pairs.

## Models compared

| Model | Type | Uncertainty | Status |
|-------|------|-------------|--------|
| **FNO** (Fourier Neural Operator) | deterministic operator | deep ensemble | Phase 1 |
| **Conditional diffusion** (U-Net backbone) | generative | sampling | Phase 1 |
| **Diffusion Transformer (DiT)** | generative | sampling | Phase 2 |
| Navier–Stokes generalization study | — | — | Phase 3 |

The point is **not** to crown a winner — FNO is a fast deterministic regressor; diffusion
models are slower but produce calibrated distributions. The deliverable is the
**accuracy / speed / uncertainty trade-off**, including how predicted uncertainty grows over
an autoregressive rollout.

## Data

- **300 flow simulations**, 24 time steps each, on a **256 × 512** grid (H × W) at 6.25 m
  spacing (a 1.6 × 3.2 km domain). Fixed injection rate and well locations; **permeability**
  is the varying input. One held-out ground-truth permeability is the showcase test case.
- Source simulations are generated with [JutulDarcy.jl](https://github.com/sintefmath/JutulDarcy.jl).
- Raw data is stored as Julia `.jld2`; `export_data.jl` converts it to portable HDF5.

Code lives in this repo; data and trained models live outside it under
`/slimdata/abhinav/neuralflow/` (see `configs/default.yaml`).

## Workflow

```bash
# 0. one-time: inspect the raw .jld2 to confirm variable names / shapes
julia export_data.jl --inspect

# 1. convert raw .jld2 -> HDF5  (writes to $data_dir)
julia export_data.jl

# 2. normalize, split by simulation, cache tensors
python prepare_data.py --config configs/default.yaml

# 3. train a model  (FNO deep-ensemble member, or a diffusion model)
python train.py --config configs/default.yaml --model fno            --member 0
python train.py --config configs/default.yaml --model diffusion-unet

# 4. evaluate: single-step + autoregressive rollout metrics
python evaluate.py --config configs/default.yaml --model fno

# 5. full 24-step rollout with uncertainty
python rollout.py --config configs/default.yaml --model diffusion-unet --samples 32
```

## Repository layout

```
neural-flow-surrogates/
├── export_data.jl        # Julia: raw .jld2  -> HDF5
├── prepare_data.py       # HDF5 -> normalized, simulation-level splits
├── train.py              # train FNO / diffusion models
├── evaluate.py           # accuracy, speed, calibration metrics
├── rollout.py            # autoregressive rollout + UQ
├── configs/default.yaml  # all paths and hyperparameters
├── neuralflow/           # importable package (models, data, metrics)
└── tests/                # shape / smoke tests (run on CPU, no data needed)
```

## Metrics

- **Accuracy** — relative L2 error, per-timestep and over the full rollout.
- **Speed** — surrogate inference wall-clock vs. the JutulDarcy reference.
- **Uncertainty** — CRPS, central-interval coverage, reliability diagrams; whether
  predicted spread grows with rollout error.
- **Front fidelity** — plume-mask IoU and front-position error (FNO spectral methods tend
  to over-smooth sharp saturation fronts — this quantifies it).

## Status

Phase 1 (FNO + conditional diffusion) is in progress. Phase 2 will add the DiT backbone;
Phase 3 will add a Navier–Stokes generalization study.

### Current results — held-out ground-truth permeability

| Model | Sat rollout rel-L2 ↓ | Pres rollout rel-L2 ↓ | Plume IoU (S > 0.1) ↑ |
|-------|----------------------|------------------------|------------------------|
| FNO (deep ensemble, 1 member) | **0.80** | 0.087 | **0.49** |
| Conditional diffusion (U-Net) | *inference bug under investigation* | — | — |

FNO training uses a combination of tricks that matter for this small-data, sharp-front
regime: per-channel residual prediction (full sat / residual pres), K-step pushforward
curriculum (Brandstetter et al., 2022), a front-aware Huber loss with a dilated plume mask,
and input-noise augmentation on the conditioning state. Together they take the sat rollout
from ~0.99 (single-step baseline) down to 0.80 with the same architecture and dataset.
See [NOTES.md](NOTES.md) for the full design rationale and failure-mode catalogue.

The current FNO rollout is not yet at the target front fidelity, and the diffusion-unet
rollout has a separate inference plumbing bug to fix — these are the next two items
on the roadmap before publishing the full accuracy / speed / uncertainty trade-off.

## License

MIT — see [LICENSE](LICENSE).

## Author

Abhinav Prakash Gahlot · [github.com/apgahlot](https://github.com/apgahlot)
