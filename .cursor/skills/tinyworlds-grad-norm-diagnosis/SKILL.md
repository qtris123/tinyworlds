---
name: tinyworlds-grad-norm-diagnosis
description: Interpret and diagnose train/grad_norm, train/grad_clipped, train/grad_skipped, and loss-spike or loss-collapse events in TinyWorlds dynamics / LAM / video-tokenizer training. Use when reading those wandb metrics, investigating sudden grad-norm bursts, mid-training loss jumps, post-LN instabilities, AdamW preconditioner decoupling symptoms, bf16 attention numerical issues, FiLM-induced amplification, or when planning a resume from a catastrophic spike (which checkpoint to load, whether to drop optimizer state, what β2 / clip / QK-Norm changes are warranted).
---

# TinyWorlds — Grad-Norm Interpretation & Spike Diagnosis

This skill captures the architectural reasons the TinyWorlds dynamics model produces transient `train/grad_norm` bursts, when those bursts are benign vs. early warnings of a catastrophic collapse, and the concrete checks + recovery moves that have already been earned in scars in this repo. All paths are relative to `tinyworlds/`.

## 1. What the three metrics actually log

All grad-norm metrics are emitted by `clip_and_log_grad_norm` (`utils/wandb_utils.py:62-90`), called once per optimizer step from each stage trainer (e.g. `scripts/train_dynamics.py:270-275`). The function calls `torch.nn.utils.clip_grad_norm_(parameters, max_norm)` and logs three series:

| Series                 | Meaning                                                                                                                                                  |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `train/grad_norm`      | **Pre-clip** global L2 norm over all `unwrap_model(model).parameters()` for that step.                                                                  |
| `train/grad_clipped`   | `1.0` when the pre-clip norm exceeded `max_norm` (and was rescaled down); `0.0` otherwise.                                                              |
| `train/grad_skipped`   | `1.0` when the pre-clip norm was NaN/Inf — the trainer then zeroes grads and **does not step** the optimizer (see `scripts/train_dynamics.py:277-284`). |

The clip threshold is hard-coded to `max_norm=1.0` in every trainer (dynamics, LAM, video-tokenizer). Grad clip operates on the DDP-all-reduced gradient (per-rank grads are already averaged before clipping), so the clip is consistent across ranks.

## 2. Healthy patterns to recognize first

| Pattern                                                | Reading                                                                                                                          |
|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Early bursts of 3–6, `grad_clipped=1` for ~500 steps   | LR warmup + AdamW second-moment EMA still building. Clip is doing its intended job. **Benign.**                                  |
| Rapid decay to ~0.2–0.5 by ~1.5 k–2 k steps            | Loss is closing on a local optimum; gradient magnitude shrinks with loss. **Normal.**                                            |
| Late-training plateau 0.1–0.4, `grad_clipped=0`        | Fine-convergence regime; clip is a passive safety net.                                                                           |
| Sporadic narrow spikes (1–3 steps) that hit `~5–6`     | **Near-misses** of the Adam-spike mechanism in §4. Worrying only if any of the §3 unhealthy signals also appear at the same step. |
| `grad_skipped` flat at 0 over the whole run            | bf16 AMP is numerically clean. The single most important guard.                                                                  |

## 3. Unhealthy patterns and the diagnosis they imply

| Pattern                                                                                  | Likely diagnosis                                                                                                                                                |
|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `grad_skipped` fires at any step after warmup                                            | AMP overflow — usually attention-logit growth past bf16 dynamic range. Check QK-Norm γ magnitudes (§5).                                                          |
| `grad_norm` spike **+ loss spike at the same step that doesn't recover**                  | The destructive part of the step was not absorbed by the clip. AdamW `v` is now poisoned; this becomes a collapse over the next few hundred steps. See §6/§7.    |
| `grad_norm` smoothly creeps up over thousands of steps                                   | Gradient heterogeneity has worsened — likely Post-LN amplification (§5.B). Inspect per-block parameter norms.                                                    |
| Loss jumps from `~0.1` → `~0.5–0.9` and **stays** at the higher plateau                  | Attention-entropy collapse / mode collapse. Optimizer has parked in a bad basin and gradients there point inward. Recovery requires reset (§7).                  |
| Spikes cluster at fixed step periods (e.g. every `N` ≈ epoch length)                     | Same hard batch reappears every epoch because `DistributedSampler.set_epoch()` is never called from the train loop (`datasets/data_utils.py:149-167`).           |

## 4. Why the spikes happen here — the Adam preconditioner decoupling story

The trigger is well-described in Wang et al. 2025, *Adaptive Preconditioners Trigger Loss Spikes in Adam* ([arXiv:2506.04805](https://arxiv.org/html/2506.04805v2)):

1. Some parameter block has a quiet stretch → AdamW's `v_t` decays geometrically by `β2^N`.
2. A batch with a moderate gradient on that block arrives.
3. Because `v` decoupled from `g²`, the effective curvature of the preconditioned Hessian briefly exceeds the stability threshold `2/η`.
4. The per-coordinate update on that block is large → global L2 grad-norm spikes.
5. Without protection, the step lands; `v` is then *poisoned* (it absorbed a giant `g²`) and stays small/wrong for `~1/(1-β2)` steps. During that window the model drifts into a worse basin.

The codebase's chosen mitigation is `adam_beta2=0.95` (see `configs/dynamics_large_lam.yaml:35-37` and the comment in `configs/dynamics.yaml:20-24`): half-life on `v` becomes ~14 steps, so the optimizer re-tracks the new gradient distribution within ~20 steps and a single spike doesn't compound. This is also the value used by [Kingma 2024 / the LLM-stability literature](https://openreview.net/pdf?id=kjSBaukyRT) for the same reason.

## 5. Four architecture-specific accelerants this codebase has

These are the levers that make the §4 mechanism *more likely to fire* in TinyWorlds dynamics than in vanilla LLM training. Worth knowing because any future block swap (e.g. Mamba-3) inherits them unless you change them.

### A. Manual bf16 softmax-attention + QK-Norm

`SpatialAttention` and `TemporalAttention` (`models/st_transformer.py:31-57, 79-111`) compute `softmax(qk^T / sqrt(d))` directly — not `F.scaled_dot_product_attention`. QK-Norm (`models/st_transformer.py:26-27, 73-74`) is what keeps q·k bounded in bf16: with `RMSNorm(head_dim)` applied to q and k, raw logits sit roughly in `[-1, +1]` after the `sqrt(d)` divide → softmax cannot saturate. This is the load-bearing defense against attention-logit growth (Dehghani et al. 2023; Wortsman et al. 2023). Without it, the first big spike usually turns into a `grad_skipped=1` event (bf16 NaN through saturated softmax).

**Caveat**: `q_norm.weight` and `k_norm.weight` (the learnable RMSNorm γ) are unbounded. If any block's `max(|γ_q|, |γ_k|)` drifts past ~2–3 over training, you've re-introduced the very problem QK-Norm was added to remove. Liu et al. 2025 ([Variance Sensitivity Induces Attention Entropy Collapse](https://aclanthology.org/2025.emnlp-main.421.pdf)). Gemma-2 composes QK-Norm with a `tanh` logit soft-cap for exactly this reason.

### B. Post-Norm with 18 blocks

Every sublayer in `STTransformerBlock` does `out = self.norm(x + sublayer_out, conditioning)` — **norm after the residual** (`models/st_transformer.py:54-55, 108-110, 126-127`). Kosson & Jaggi 2025 ([Understanding Transformer Optimization via Gradient Heterogeneity](https://arxiv.org/html/2502.00213v3)) prove that in Post-LN the LayerNorm Jacobian scales the *entire* upstream Jacobian multiplicatively in the backward pass, whereas in Pre-LN it only scales the residual branch. The empirical consequence: per-block gradient norms in Post-LN are far more heterogeneous than in Pre-LN. With 18 stacked blocks, that heterogeneity directly feeds the §4 decoupling mechanism (different blocks reach the "quiet stretch" at different times).

### C. FiLM modulation with unbounded γ and missing weight decay

`AdaptiveNormalizer` (`models/norms.py:30-72`) is `SimpleLayerNorm(x) * (1 + gamma) + beta`, where `gamma, beta` come from a `Linear(conditioning_dim, 2*embed_dim)` over the (frozen) LAM action latents. Two compounding subtleties:

- `SimpleLayerNorm` (`models/norms.py:18-27`) is **z-score only — no learnable scale**. So `(1 + gamma)` is the *only* scale on the residual stream. If `gamma` ever takes a large unsigned value on any token, the residual stream is amplified through that block and the next block's gradient is correspondingly heterogeneous.
- The `_split_decay_params` heuristic (`utils/optimizer_utils.py:78-87`) puts any parameter whose name path contains `"norm"` into the `no_decay` group. `AdaptiveNormalizer.to_gamma_beta` is buried inside paths like `…ffn.norm.to_gamma_beta.…`, so its Linear weights receive **no weight decay**. They drift freely over training.

Init is `std=1e-3` so this is benign early. It is the slowest-moving accelerant — worth checking only on long runs.

### D. MaskGIT random per-batch mask ratio in `[0.5, 1.0)`

`DynamicsModel.forward` (`models/dynamics.py:44-52`) samples a fresh mask ratio per forward call. With 4 GPUs × `gradient_accumulation_steps=4` that's 16 independent samples per optimizer step; ~60 % of steps have at least one micro-batch with mask ratio > 0.95 (predict ~all tokens from a single anchor). The masked cross-entropy denominator is the count of masked positions so the *scalar* loss stays bounded, but the **per-block gradient distribution** changes dramatically with mask ratio. This is extra variance for AdamW's `v` to track, multiplying the chance of a §4 decoupling event.

## 6. Why the current run survives the same trigger

The combination of three independent defenses (added after the spiked run in §7):

| Defense                                              | Where                                                                  | What it absorbs                                                                          |
|------------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| QK-Norm on q and k                                   | `models/st_transformer.py:26-27, 40-41, 73-74, 88-89`                  | Attention-logit growth → bf16 softmax saturation → NaN                                    |
| `adam_beta2=0.95`                                    | `configs/dynamics_large_lam.yaml:35-37`, `configs/dynamics.yaml:22-24` | Adam preconditioner staying poisoned (§4); recovery ~20 steps instead of ~1000            |
| `clip_grad_norm_(…, max_norm=1.0)` + non-finite skip | `scripts/train_dynamics.py:270-284`                                    | The destructive part of a single spike step; AMP overflow batches no longer corrupt `v` |

When all three are present, the §4 mechanism still fires — that's the narrow spikes you see in `train/grad_clipped` mid-training — but each spike is reduced to a 1–3 step blip that doesn't visibly move `train/loss`.

## 7. The prior collapse — case study `2026_06_21_13_02_14`

Recorded permanently in `configs/dynamics.yaml:1-5`:

> Training — d836767 codebase + QK-Norm + AdamW β2=0.95 + grad-norm logging
> Resumed from step-8500 of run 2026_06_21_13_02_14 (last healthy checkpoint
> before the loss-spike at step 8.8k). Optimizer state is intentionally NOT
> loaded — the saved exp_avg_sq was poisoned (see investigation notes).

What that run had vs. what was added:

| Aspect                | `2026_06_21_13_02_14` (collapsed)                            | Current recipe                                  |
|-----------------------|--------------------------------------------------------------|-------------------------------------------------|
| QK-Norm               | **absent**                                                   | present                                         |
| `adam_beta2`          | higher (post-mortem: `v` took ~1000 steps to recover)        | `0.95`                                          |
| Grad clip + NaN skip  | not present / not logged                                     | `max_norm=1.0`, skip + zero on non-finite       |
| Visible failure       | `train/loss` jumped from ~0.1 → ~0.85 at step ~8.7 k and **stayed flat at the new plateau** | smooth                                          |

The visible failure shape (a vertical jump to a higher, *stable* plateau) is the textbook signature of attention-entropy collapse — the optimizer overshot, `v` was poisoned, the model drifted into a basin where attention rows became near-one-hot or near-uniform, and the loss landscape at the new basin no longer pulls back toward the old one.

## 8. Diagnostic checklist when investigating a new event

Walk this list in order before changing any hyperparameter:

1. **Is `train/grad_skipped` non-zero anywhere?** If yes — AMP overflow happened. Look at the closest grad-norm spike; the cause is almost always attention-logit growth (§5.A). Action: log `q_norm.weight.abs().max()` / `k_norm.weight.abs().max()` per block and inspect.
2. **Did `train/loss` move on the same step as the grad spike?** Cross-check with the `Step N Loss` lines in the run's stdout log (`logs/dyn_*.log`).
   - Loss unchanged → near-miss; mechanism §4, defenses §6 worked. No action.
   - Loss bumped up but recovered within ~50 steps → spike was partly absorbed; `v` recovered (β2=0.95 working). No action.
   - **Loss bumped up and stayed up** → collapse. Stop the run; jump to §9.
3. **Compare `train/loss` and `eval/loss`.** Eval loss should track train loss within a small constant. If `eval/loss` is many ×  train loss and *increasing*, you are overfitting hard, not unstable. Different problem; not what this skill addresses.
4. **Look for spike periodicity.** If grad-norm spikes recur at intervals close to `len(train_data) / (world_size * batch_size_per_gpu * grad_accum)`, it's the missing `DistributedSampler.set_epoch()` (`datasets/data_utils.py:149-167`). Add `train_sampler.set_epoch(epoch_idx)` at the iterator reset point in the trainer.
5. **For long runs, sample γ magnitudes.** Add (cheap) wandb logging of `model.transformer.blocks[i].spatial_attn.q_norm.weight.abs().max()` and `.k_norm.weight.abs().max()` per block, every `log_interval` steps. Drift past ~2–3 = QK-Norm losing its grip; consider adding a Gemma-2 style soft-cap on logits.

## 9. Recovery playbook after a real collapse

If §8.2 puts you in the "loss bumped up and stayed up" case:

1. **Find the last healthy checkpoint.** TinyWorlds saves dynamics checkpoints every `log_interval` to `results/<timestamp>/dynamics/checkpoints/dynamics_step_<N>/`. Walk backwards from the collapse step.
2. **Resume model weights only — never the optimizer.** Use `load_dynamics_from_checkpoint(..., strict=False)` (`scripts/train_dynamics.py:93-102`). The `strict=False` is important for adding stability params (e.g. QK-Norm) on top of an older architecture; the optimizer state must be left freshly initialized because `exp_avg_sq` from before the spike is poisoned (`configs/dynamics.yaml:1-5` is the precedent).
3. **Add at least one defense that was missing** before relaunching. The minimum bar is:
   - QK-Norm present in every attention sublayer.
   - `adam_beta2 ≤ 0.95`.
   - `clip_grad_norm_` with `max_norm ≤ 1.0` and non-finite skip wired in.
4. **Keep the prior run's logs and checkpoint metadata.** The repo's habit of leaving an explanatory comment block at the top of the config (`configs/dynamics.yaml:1-5`) is the convention to extend: every recovery config should encode *which run failed*, *which step it failed at*, and *which defense the new config adds*.

## 10. Quick-reference checklist for adding new training defenses

If you're about to add another stabilizer, in roughly increasing cost:

- [ ] Lower LR or extend warmup (`utils/scheduler_utils.py` — `warmup_fraction` default 0.05). Cheap; usually enough by itself for borderline cases.
- [ ] Lower `adam_beta2` further (0.9 is the next step from 0.95).
- [ ] Add Gemma-2 style logit soft-cap (`logits = soft_cap * tanh(logits / soft_cap)`) inside `SpatialAttention` / `TemporalAttention` after the `q@k.T / sqrt(d)` line and before `F.softmax`. ~5 lines of code; documented in [arXiv:2408.00118](https://arxiv.org/html/2408.00118).
- [ ] Switch to Pre-LN (move `AdaptiveNormalizer` to *before* the sublayer call and drop it after the residual). Larger change; revisit gradient heterogeneity.
- [ ] Apply weight decay to `to_gamma_beta` by tightening `_split_decay_params` (`utils/optimizer_utils.py:83`) so the `"norm"` substring rule doesn't accidentally exempt FiLM projection weights.

## References

- Wang et al. 2025, *Adaptive Preconditioners Trigger Loss Spikes in Adam* — [arXiv:2506.04805](https://arxiv.org/html/2506.04805v2). Root cause of the §4 mechanism.
- Kosson & Jaggi 2025, *Understanding Transformer Optimization via Gradient Heterogeneity* — [arXiv:2502.00213](https://arxiv.org/html/2502.00213v3). Post-LN vs Pre-LN gradient analysis.
- Dehghani et al. 2023 (ViT-22B), Wortsman et al. 2023, *Small-scale proxies for large-scale Transformer training instabilities* — [arXiv:2309.14322](https://ar5iv.labs.arxiv.org/html/2309.14322). QK-Norm origin and the attention-logit-growth failure mode.
- Liu et al. 2025, *Variance Sensitivity Induces Attention Entropy Collapse in Transformers* — [aclanthology.org/2025.emnlp-main.421](https://aclanthology.org/2025.emnlp-main.421.pdf). Why QK-Norm alone isn't a complete fix.
- Gemma-2 — [arXiv:2408.00118](https://arxiv.org/html/2408.00118). Logit soft-cap as a complementary defense.
