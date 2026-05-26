# Neural Flow Surrogates — Design Notes

A self-contained reference describing **what** each model is, **why** the
training objective looks the way it does, **what failed before** each fix went
in, and **how the same fixes apply** to the diffusion models when we get to
them.

The codebase trains autoregressive surrogates that emulate one timestep of
JutulDarcy two-phase CO₂ flow on a 256 × 512 grid: given a permeability field
`K` and the joint state `(S_t, P_t)`, predict `(S_{t+1}, P_{t+1})`. Forecast
trajectories are produced by chaining 23 such steps. The benchmark is a
held-out ground-truth (GT) permeability case.

---

## 1. Problem framing

The surrogate is a discrete-time operator
$\mathcal{F}_\theta : (K, S_t, P_t) \mapsto (S_{t+1}, P_{t+1})$
trained on next-step transitions. At inference we **roll out** this map for 23
steps to produce the full trajectory.

Two failure modes dominate this setup, and the four design changes below each
target one of them:

| Failure mode | Symptom | Fixes that address it |
|---|---|---|
| **Sparse target (plume in zero background)** | Loss minimized by predicting "no plume"; sat rel-L2 stays high even at low training loss. | Channel weighting, front-aware Huber. |
| **Train/test distribution shift (compounding error)** | Per-step val MSE tiny (~10⁻³), 23-step rollout rel-L2 huge (~0.8). Per-step error curve climbs over the horizon. | Residual prediction, unrolled training (pushforward), curriculum. |

Both failures are *invisible* at the per-step val-loss level. The diagnostic
tools that surface them are (a) per-step rollout rel-L2, (b) plume IoU at
`S > 0.1`, and (c) the rollout figure itself.

---

## 2. Models

All three models share the conditioning interface: a 3-channel input
`[K, S_t, P_t]` produces a 2-channel output `[S_{t+1}, P_{t+1}]`.

### 2.1 FNO2d (`neuralflow/models/fno.py`)

Fourier Neural Operator (Li et al., *ICLR 2021*). Acts in spectral space: each
layer applies a learned linear transform on the lowest `(modes1, modes2)`
Fourier modes of the field, plus a pointwise 1×1 convolution residual path.

Why FNO for this problem:
- **Global receptive field per layer.** Plume migration is a transport process —
  information propagates across the whole field every step. CNNs need many
  layers to do the same; FNO does it in one.
- **Resolution agnostic.** The Fourier basis is grid-independent, so the
  trained model transfers across grid sizes.
- **Cheap.** O(N log N) per layer (FFT) vs O(N²) for a transformer at this
  spatial size.

Architecture (`width=36, modes=(32,32), n_layers=4`):

```
cond (B, 3, H, W) ──┐
coord grid (B, 2, H, W) ──┐
                          ├─► concat ─► lift Conv1x1 → 36 ch
4× residual block:
    x  ←  x + GELU(GroupNorm( SpectralConv2d(x) + PointwiseConv2d(x) ))
                          ─► proj: 1x1 → 128 → GELU → 1x1 → 2
output: delta or state (see § 3.2)
```

Parameter count: 21.24 M. (Half of these live in the complex spectral weights
stored as real (..., 2) tensors — see the GradScaler-fix block in
`SpectralConv2d.__init__`. Counts of complex tensors give "DOFs"; counts of the
underlying real storage give "trainable floats". Same memory, same model.)

Uncertainty: obtained via a **deep ensemble** of M=5 independently initialized
FNOs. Each member is trained with its own seed (`--member 0..4`). At inference
we average their rollouts and use the spread as epistemic uncertainty.
*Caveat:* ensembles average decorrelated errors; they do **not** fix shared
biases (e.g., a wrong inductive bias from the loss function). They are
correctly thought of as UQ + variance reduction, not as a primary error fix.

### 2.2 Conditional U-Net diffusion (`neuralflow/models/unet.py`, Phase 2)

A standard `GaussianDiffusion(denoiser)` wrapper around a `CondUNet`. The
denoiser takes `(x_t, cond, timestep)` where `x_t` is the noised next-state and
`cond = [K, S_t, P_t]` is concatenated channel-wise. Training is
ε-prediction with a cosine noise schedule; inference uses DDIM (50 steps by
default).

Why diffusion in addition to FNO:
- **Generative UQ.** Each call samples a different trajectory. With ~32 samples
  per step we get a per-pixel posterior over plume position — qualitatively
  different from the ensemble spread of a deterministic surrogate.
- **Stochasticity at the front.** The plume front position is the most
  uncertain quantity; a generative model represents that uncertainty in the
  output distribution, not just as an epistemic spread.

The UNet itself: base channels = 64, mults = `[1, 2, 2, 4]`, self-attention at
the bottleneck only. Standard.

### 2.3 Diffusion Transformer (`neuralflow/models/dit.py`, Phase 2)

Same diffusion wrapper as 2.2, but the denoiser is a **DiT** (Peebles & Xie,
*ICCV 2023*) — patch the field into 16×16 patches, run a transformer over the
token sequence with AdaLN conditioning on `(t, cond)`. `hidden_size=384,
depth=8, heads=6`.

Why DiT alongside UNet:
- **Long-range conditioning on K.** The permeability field has long-range
  geological structures (channels, layers); attention can model this without
  bottlenecking through downsampling layers.
- **Cleaner scaling.** DiT scales by depth/width without architectural surgery,
  useful when we want a strong baseline.

DiT is the most expensive of the three and the last we'll train.

---

## 3. The four training-objective changes

Each subsection states the change, the failure mode it addresses, what we
tried before, and where it's implemented.

### 3.1 Front-aware Huber saturation loss

**The change.** Replace plain MSE on saturation with smooth-L1 (Huber) over the
field, plus an extra multiplicative weight on cells inside the (dilated)
ground-truth plume:

```
mask  = dilate( target_sat > sat_threshold, radius = mask_dilate )
sat_l = ((1 + alpha * mask) * smooth_l1(pred_sat, target_sat, beta)).mean()
loss  = sat_w * sat_l  +  pres_w * mse(pred_pres, target_pres)
```

Implementation: `train.py:_front_aware_huber` and `_dilate_mask`. Hyperparams
in `configs/default.yaml::train`: `sat_threshold=0.01`, `mask_alpha=20`,
`mask_dilate=3`, `huber_beta=0.05`.

**Why it works.**
- The plume occupies ≈5–10 % of the 256×512 field. Even with the existing
  channel weight `sat_loss_weight=10` (saturation weighted 10× pressure), the
  spatial average of MSE inside the saturation channel is itself dominated by
  the vast `S=0` background. The model satisfies the loss by predicting "no
  plume" everywhere and is rewarded for it.
- The mask up-weights plume cells by `(1 + α)` ≈ 21× on top of the channel
  weight. Now the model's gradient share inside the plume is comparable to
  outside.
- **Dilating** the mask by 3 cells creates a leading-edge halo that includes
  cells that *should become wet next step* but currently aren't. Without
  dilation, the loss has no signal about where the front *should* move to.
- **Huber over L1:** pure L1 has a discontinuous gradient at zero, which is bad
  when the target is exactly zero almost everywhere. Smooth-L1 has a quadratic
  region near zero (controlled by `beta=0.05` in normalized units, ≈ `0.023` in
  physical saturation units) and linear tails — best of both worlds.

**Pressure stays on MSE** because it's a smooth, globally-coupled field;
weighting and L1 don't help, and unweighted MSE already gives `rel-L2 ≈ 0.017`.

**What we tried before.**
- Plain MSE on both channels (default): saturation rollout rel-L2 = 0.99
  (catastrophic identity-collapse-like behavior).
- Channel-weighted MSE with `sat_loss_weight=10`: saturation rel-L2 = 0.78.
  Better, but still dominated by background; plume shape wrong.
- Front-aware Huber (this change) is the next step.

**Applies to diffusion models?** Yes, with one adaptation. Diffusion models
predict noise ε, so the natural loss is `||ε - ε̂||²`. Two options:
1. **Apply the mask to the x₀-prediction reparameterization.** With
   `v`-parameterization or x₀-parameterization, the loss can be written on the
   clean image; mask the clean-image loss the same way as for FNO.
2. **Apply the mask in image space at inference time** (rejection-style sample
   filtering against a posterior front constraint). Cheaper but weaker.
   We will use option 1.

---

### 3.2 Residual (delta) prediction

**The change.** The FNO's projection head outputs an increment
`delta = (delta_S, delta_P)`, and the model returns `state_t + delta`:

```python
delta = self.proj(features)               # raw conv output
state_t = cond[:, 1:1 + self.out_ch]      # [S_t, P_t] from conditioning
return state_t + delta                    # if self.residual else delta
```

Implementation: `FNO2d.__init__(residual=True)` in
`neuralflow/models/fno.py`. The clamp into the physical range `[-1, 1]` (in
normalized units; physical `[0, 0.9]` for saturation) is done **outside the
model** in `neuralflow/inference.py::rollout_fno` — that lets the loss see
unclamped predictions during training (so the gradient pulls back from
out-of-range values) and clamps only during autoregressive rollout.

**Why it works.**
- Physically, saturation evolves as `S_{t+1} = S_t - ∇·(v·S) Δt`. The
  *increment* is small and local; the *state* is a globally-coupled field with
  a sharp front. Asking the model to regenerate the whole field every step is a
  harder learning problem than asking it to predict the increment.
- The output distribution of `delta` is centered near zero with small
  magnitude, so the model's effective output scale is tiny — easier
  optimization, smaller per-step error magnitude.
- Reduces (does **not** eliminate) compounding: by step k>0, `S_t` in the
  residual sum is itself a model prediction, so errors still accumulate. But
  per-step error magnitude is much smaller, giving compounding less to grow
  from. Composes cleanly with the pushforward trick (§ 3.3).
- The residual head is **applied per channel** (`residual_per_channel:
  [false, true]` for `[sat, pres]`). Saturation needed it OFF — see below.

**Per-channel revision (rev 2).** After 100 epochs the residual head fit a
near-zero delta prior for saturation: val_sat held at 0.045 while train_sat
dropped to 0.0056 (8× gap), and rollout rel-L2 stayed at 0.88. Diagnosis:
saturation deltas near the front are *large* at early times (front sweeps
into new cells), but the `state_t + delta` parameterization biases the head
toward small corrections everywhere — the model "wins" by predicting near
zero and accepting front lag. Plume IoU was 0.92 single-step (location
right) while rel-L2 stayed high (values wrong). For saturation we now
predict the full field directly; for pressure (smooth, slow, val_pres ≈
1.5e-5 already excellent) the residual prior is correct and we keep it.

**Why no clamp inside the model:**
Hard clamping inside `forward()` zeroes the gradient on out-of-range
predictions, which prevents the model from learning to pull back into range.
The MSE loss already penalizes out-of-range predictions; the inference-time
clamp in `rollout_fno` handles rollout stability. Best of both worlds.

**What we tried before.**
- Direct state prediction (no residual): the head had to produce the full
  field at every step. With channel-weighted MSE, this gave sat rollout = 0.78.

**Applies to diffusion models?** *Partially.* Diffusion already operates on
residuals — ε-prediction is literally noise, which is the residual between
`x_t` and the clean image scaled by the schedule. The cleaner analog is:
- **Predict `Δx = x_{t+1} - x_t` instead of `x_{t+1}`** as the diffusion
  target. The diffusion process then runs over a small-magnitude delta field
  rather than the full state, which empirically converges faster and yields
  more stable rollouts (similar reasoning to FNO). Implementation: set
  `target = state_{t+1} - state_t` in the diffusion training loop, sample
  `delta_pred ~ p_θ(delta | cond)` at inference, and return
  `clamp(state_t + delta_pred)`. We will add this when we start the diffusion
  training.

---

### 3.3 Unrolled training (the pushforward trick)

**The change.** During training, each batch is a K-step window from a single
simulation. The model is rolled out K times within the training step, with its
own prediction fed back as the next conditioning input. The intermediate
prediction is `.detach()`-ed so the gradient flows only through the **current**
step's network call.

```python
cond = batch["cond"]                       # (B, 3, H, W) at t=0
for step_idx in range(k_unroll):
    pred = model(cond)                     # (B, 2, H, W)
    loss_k = sat_w * front_huber(pred_sat, targets[:, k, 0:1]) \
           + pres_w * mse(pred_pres, targets[:, k, 1:2])
    losses.append(loss_k)
    if step_idx < k_unroll - 1:
        cond = torch.cat([perm, pred.detach()], dim=1)   # pushforward
total = stack(losses).mean()
```

Implementation: `train.py::_fno_step`. Dataset:
`FlowSequenceDataset(k_steps=K)` in `neuralflow/data.py` returns `(cond,
targets, perm)` with `targets` of shape `(K, 2, H, W)`.

**Why it works.**
- Teacher-forced training shows the model only ground-truth inputs. At
  inference, by step k>0 the input is the model's own (slightly wrong)
  prediction — a distribution it never saw during training. This is **the**
  source of compounding error.
- Unrolled training makes the model see its own outputs during training,
  closing the train/test distribution gap.
- **The `.detach()` is load-bearing.** It (a) keeps memory bounded — without it
  you backprop through K nested FNOs (K=2 → 2× memory through the spectral
  weights); (b) makes optimization stable on small datasets; (c) keeps the
  per-step loss meaningful — the gradient at step k+1 is "given a corrupted
  input that looks like a model prediction, produce the right output", which is
  exactly the inference-time task.

**Why K=2 (and curriculum to it):**
- Brandstetter et al. (*Message Passing Neural PDE Solvers*, ICLR 2022) tested
  K=1..8 on advection-dominated PDEs and found K=2 captures **most** of the
  rollout-error reduction; K=3–4 gives diminishing returns at 1.5–2× compute.
- For our 23-step horizon with small data (5,175 transitions), K=3 risks
  overfitting to the larger effective batch.
- Cost: ≈ 2× per-epoch wall time vs single-step training. With val plateau at
  ~epoch 20, 40 epochs of K=2 training fits one overnight slot.
- **Escalation criterion** if rollout still bad after retrain: per-step rel-L2
  trajectory tells you whether to (a) bump K → 3 (if sharp climb after step
  10–15), (b) tune Huber α (linear climb from step 1), or (c) accept the
  result (flat or gently-sloped curve).

**Rev 3 — input-noise augmentation.** A *generalization* of the pushforward
trick: at every training step, add small Gaussian noise to the state portion
of the conditioning before the forward pass (`input_noise_sat`,
`input_noise_pres` in YAML; default 0.03 / 0). Why: pushforward only exposes
the model to *its own* prediction errors during the K-1 unrolled steps —
but those errors are correlated with the network's biases. Random noise
samples a broader set of perturbations the model must recover from, which
empirically tightens the gap between single-step val and autoregressive
rollout. Active only during training; `model.eval()` makes the helper a
no-op, so val metrics remain a fair single-step signal. Magnitude tuning:
~⅔ of single-step val_sat is a reasonable starting point (small enough not
to dominate the signal, large enough to matter).

**Run 3 result (σ_sat = 0.03, 100 epochs):** sat rollout 0.8713 → 0.8371,
plume IoU 0.46 → 0.48, train/val ratio 8× → 2.6× → **3×** (closed). The
single-step val_sat floor *rose* from 0.045 to 0.060 — expected, since
noise-aug intentionally trades a little single-step accuracy for robustness.
**Run 4 (σ_sat = 0.05, in flight):** bumping noise to the level of the
single-step floor itself. If sat rollout still > 0.7, the bottleneck is data
volume (225 train sims, 23 transitions), not training objective — switch
focus to diffusion-unet, which can amortize the prior better at the same
data budget.

**What we tried before.**
- Single-step training (default): saturation rollout = 0.99 → 0.78 with
  channel weighting alone.
- 100-epoch single-step (overfitting confirmed at epoch ~20): no improvement
  beyond the 10-epoch result (val loss flat at 0.008, but train loss went 50×
  lower → 10× train/val gap → overfit).
- Unrolled training (this change) is the next step.

**Applies to diffusion models?** *Yes, but the implementation differs.* Doing
full unrolled training for diffusion is prohibitive (each step requires a 50-
step sampling chain inside the training loop). Two cheaper analogs:
1. **Scheduled sampling.** With probability `p_schedule` (annealed from 0 to
   ≈0.5 over training), replace the ground-truth `S_t, P_t` in the conditioning
   with a *cheap* sample from the model (10-step DDIM, one shot, no
   backprop). The model still sees corrupted inputs without ever unrolling
   gradients through sampling. This is the standard fix in autoregressive
   generative modeling (originally Bengio et al. 2015 for RNNs).
2. **Single-step pushforward via x₀-prediction.** Reparameterize the loss to
   predict the clean next-state; treat the previous step's clean prediction (no
   gradient) as the conditioning. Cheaper still; matches the FNO recipe.
   We will start with (2) and only escalate to (1) if rollouts compound.

---

### 3.4 Curriculum unrolling

**The change.** Train with K=1 for the first `warmup_k1_epochs` epochs (default
5), then switch to K=k_steps_max (default 2) for the rest. Implementation:
`train.py::_k_for_epoch`.

**Why.**
- At random init, the model's K=2 prediction is *very* wrong — feeding garbage
  back as input gives a noisy loss that can stall optimization on small data.
- A short K=1 warmup gets the model into a basin where its predictions are
  approximately correct; from there K=2 finetunes for distribution-shift
  robustness.
- Costs nothing — warmup is just fewer forward passes per step.

**What we tried before.** No curriculum. Brandstetter et al. reported similar
benefits from a "one-step warmup" on small PDE datasets.

**Applies to diffusion models?** Yes, analogously: train without scheduled
sampling for a few epochs (pure teacher forcing on noise prediction), then ramp
the scheduled-sampling probability from 0 to 0.5.

---

## 4. Hyperparameter table

| Knob (in `configs/default.yaml`) | Default | What it controls | Tune if ... |
|---|---|---|---|
| `train.sat_loss_weight` | 10 | Channel weight S over P | Pressure rel-L2 starts to drift (lower it) |
| `train.pres_loss_weight` | 1 | Channel weight on pressure | Always 1; vary sat_w instead |
| `train.sat_threshold` | 0.01 | Physical S above which a cell counts as "plume" | Mask too small/large (raise/lower) |
| `train.mask_alpha` | 20 | Extra weight on plume cells in Huber | Background drifts (lower to 10); plume undershoots (raise to 30–50) |
| `train.mask_dilate` | 3 | Cells of mask dilation (leading-edge halo) | Front consistently behind GT (raise to 5) |
| `train.huber_beta` | 0.05 | Smooth-L1 transition (normalized sat units) | Plume shape jagged (raise to 0.1) |
| `train.k_steps_max` | 2 | Pushforward unroll depth | Per-step rel-L2 climbs sharply (raise to 3) |
| `train.warmup_k1_epochs` | 5 | K=1 warmup before K=k_max | Loss spikes when K switches (raise to 8) |
| `train.weight_decay` | 1e-4 | AdamW weight decay | Train/val gap >5× (raise to 3e-4) |
| `model.fno.residual_per_channel` | [false, true] (sat, pres) | Per-channel residual switch | Flip saturation back on only if front speed becomes unstable |
| `model.fno.modes` | [20, 20] | FNO spectral modes per axis | val/train >6× → drop to [16, 16]; underfitting front detail → raise |
| `eval.front_threshold` | 0.1 | Plume IoU threshold (physical S) | Match a paper / domain convention |

---

## 5. Evaluation metrics

What we track and what each one tells you:

| Metric | What it answers | Failure signature |
|---|---|---|
| Per-step val loss | Is the model fitting next-step transitions? | Tiny number that hides everything below. |
| **Rollout rel-L2 (aggregate)** | How wrong is the 23-step trajectory in L2? | Pressure ≈ 0.02 OK; saturation < 0.30 ship-quality. |
| **Rollout rel-L2 (per timestep)** | Where does the error live in time? | Flat → fixed compounding; rising → distribution shift. |
| **Plume IoU at S > 0.1** | Is the plume in the right place? | The metric that actually matters visually. |
| Train/val loss ratio | Are we overfitting? | >5× → bump weight decay or early-stop. |
| Ensemble spread (M=5) | Epistemic UQ | Large where front is (good); large in background (bad). |

`rollout.py` now prints all of these.

---

## 6. Failure-mode catalogue (history)

What we tried, what broke, why.

1. **Plain MSE, 10 epochs.** Saturation rollout rel-L2 = 0.99. *Cause:* loss
   dominated by zero background; model predicts zero everywhere.
2. **Channel-weighted MSE, 10 epochs.** Saturation rollout = 0.91. *Cause:*
   weighting fixes pressure-vs-saturation imbalance but not the
   plume-vs-background imbalance *within* the saturation channel.
3. **Channel-weighted MSE, 100 epochs.** Saturation rollout = 0.78. *Cause:*
   per-step val loss converged at epoch ~20; rest of training overfit. Best
   checkpoint = ~epoch 20 ≈ same as 10-epoch run for next-step quality but
   better rollout (settled into a flatter basin).
4. **AMP + complex spectral weights (early).** `RuntimeError: ComplexHalf`
   inside `einsum`. *Cause:* GradScaler can't unscale complex Parameters.
   *Fix:* store weights as real (..., 2) tensors, `view_as_complex` inside
   `_mul`. Cosmetic side effect: parameter count appears to double (real
   numel counts floats; complex numel counts complex numbers).
5. **K-step (K=2) + front-aware Huber + residual on both channels, 100 ep.**
   Saturation rollout = 0.90, IoU = 0.45. Best ckpt saved at K=1 phase
   (epoch 11). *Cause:* val loss climbed after epoch ~30; weight_decay 1e-4
   and mask_alpha=20 over-fit plume locations on train. *Fix:* alpha 20→10,
   wd 1e-4→3e-4, warmup_k1 5→10, dilate 3→5. (Loss-tuning regime.)
6. **Same with retuned loss/regularization, 100 ep.** Saturation rollout =
   0.88, IoU = 0.46. val_sat ↓ to 0.045 (from 0.078) but rollout didn't move
   — single-step floor became the bottleneck, not compounding. val/train
   ratio still ~8×. *Diagnosis:* (i) FNO over-capacity (1024 modes, 300
   sims) and (ii) residual head fitting near-zero deltas for saturation.
   *Fix:* drop modes [32,32]→[20,20], flip residual off for saturation
   (keep for pressure). Both target the per-step generalization floor.
   (Architectural regime — loss tuning has hit diminishing returns.)

The four changes in § 3 target the residual failure (sat = 0.78). Expected
post-fix rollout based on published numbers on similar PDE problems:

- Front-aware Huber alone: ≈ 0.50–0.60.
- + residual prediction: ≈ 0.40–0.50.
- + unrolled K=2: ≈ 0.25–0.40.
- + deep ensemble (M=5): ≈ 0.20–0.35.

These are *additive in log-space, not multiplicative*; each one chips at a
different failure mode.

---

## 7. References

- Li et al., *Fourier Neural Operator for Parametric Partial Differential
  Equations*, ICLR 2021 — base FNO architecture.
- Brandstetter, Worrall, Welling, *Message Passing Neural PDE Solvers*, ICLR
  2022 — the pushforward trick (§ 3.3) and K-step ablations.
- Lippe et al., *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE
  Solvers*, NeurIPS 2023 — confirms the diagnosis of compounding error in
  rollouts and proposes a complementary refinement step. Worth reading before
  Phase 2.
- Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023 —
  DiT architecture.
- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020 —
  the diffusion-UNet baseline.
- Bengio et al., *Scheduled Sampling for Sequence Prediction with Recurrent
  Neural Networks*, NeurIPS 2015 — the technique we'll use for the diffusion
  pushforward analog (§ 3.3, item 1).

---

## 8. What changed in this revision (file-level)

For when you read the diff later.

- `neuralflow/data.py` — added `FlowSequenceDataset` returning K-step windows;
  `FlowDataset` left untouched for the diffusion path.
- `neuralflow/models/fno.py` — `FNO2d.__init__` takes `residual` as bool *or*
  per-channel list; an `out_ch`-wide buffer mask is broadcast over the
  output so the residual prior can be on/off per physical variable. No
  clamp inside the model.
- `neuralflow/models/__init__.py` — factory reads `residual_per_channel`
  from the YAML (falls back to scalar `residual`) and threads it into
  `FNO2d`.
- `train.py` — rewritten step function: K-step unrolled loss with detach on
  intermediate predictions; front-aware Huber on saturation; pressure MSE
  unchanged; plume-IoU tracking on validation; curriculum K=1→K=k_max.
- `rollout.py` — now prints per-timestep rel-L2 (sat & pres) and plume IoU at
  the configured front threshold, in addition to the aggregate.
- `configs/default.yaml` — new keys under `train` for the four changes; bumped
  `weight_decay` 1e-5 → 1e-4 to close the train/val gap.

Untouched (intentionally): `neuralflow/inference.py` — the existing
`rollout_fno` already clamps post-model, which is what we want.
