---
name: tinyworlds-dynamics-training-observations
description: Empirical observations, debugging notes, and findings from training the TinyWorlds dynamics model on Zelda (DDP on 4× GH200). Covers the U-shaped eval-loss / overfitting investigation, the DDP checkpoint save-race bug + fix, GH200 memory budget per batch size, params-vs-data scaling for Zelda, and recommended next experiments. Use when investigating overfitting / eval-loss U-shape on the dynamics stage, picking a model size for this dataset, debugging checkpoint corruption or "1315-byte" empty model_state_dict.pt files under DDP, choosing batch size / grad-accum on GH200, or planning regularization / early-stopping experiments. For grad-norm bursts, loss spikes, or post-LN/QK-Norm instabilities see the sibling skill `tinyworlds-grad-norm-diagnosis` instead.
---

# TinyWorlds Dynamics Training — Empirical Observations

Notes from training experiments on the Zelda dataset (`data/zelda_frames.h5`, 64×64 RGB) with the dynamics stage of the TinyWorlds pipeline on 4× GH200 96 GiB with DDP. The architectural reference for the codebase lives in the sibling skill `tinyworlds-genie-worldmodel`; this skill captures empirical numbers and bug post-mortems specific to dynamics training.

All file paths are relative to the repo root (`tinyworlds/`).

## 1. The U-shaped eval-loss "spike" — overfitting, not numerical instability

### Observation

In `results/2026_06_23_09_31_58_30k`, eval loss appeared to **spike up** around step ~10k while train loss kept dropping. Initial framing was "eval loss spikes" suggesting numerical instability; closer inspection showed a **U-shape**, not a spike — the classical overfitting signature.

A re-run with the fixed checkpoint code (`results/2026_06_24_15_31_15`, `bs=32`, `grad_accum=4`, eff_batch=512, same model: 18-block / 512-dim / 2048-hidden) reproduced the U:

| Step  | Train CE | Eval CE |
|------:|---------:|--------:|
| 0     | 1.76     | 7.11    |
| 1000  | 0.69     | 3.12    |
| 7000  | 0.143    | **2.97** ← eval minimum |
| 8000  | 0.134    | 3.07    |
| 9000  | 0.127    | 3.18    |
| 10000 | 0.118    | 3.35    |
| 11000 | 0.111    | 3.39    |

Train loss is monotonic ↓ (memorization); eval loss bottoms at step ~7000 and rises thereafter.

### Diagnosis

Four contributing factors, in order of impact:

1. **Each unique token is presented ~800× across 30k steps.** With eff_batch=512, num_frames=4, patches=256/frame, mask_ratio≈0.75: per step the model sees `512 × 4 × 256 × 0.75 ≈ 393k` target tokens. Over 30k steps → ~11.8 B target-token presentations. Unique Zelda training tokens: only ~14.6M (see §4). Ratio ≈ 800×.
2. **Chronological 80/10/10 train/eval/test split** (`enable_partition=true` in `training_large_lam.yaml`). Train = first 80% of frames, eval = next 10%. Game progression introduces distribution shift between partitions — eval frames may show areas/enemies absent in train.
3. **No dropout in `models/st_transformer.py`** (neither `SpatialAttention`, `TemporalAttention`, nor `SwiGLUFFN` apply dropout). Light `weight_decay=0.01` in `utils/optimizer_utils.py`.
4. **Model over-parameterized relative to data** (see §4 for the ratio).

### What's NOT the cause

- It's **not a loss spike / NaN / optimizer instability**. Gradients, AdamW `exp_avg_sq`, and AMP-bf16 cast behavior are all healthy. The recipe `adam_beta2=0.95` (chosen post-mortem on a prior spike) is keeping any actual numerical wobble bounded. If a future run looks superficially similar but `train/loss` itself jumps and stays elevated, that's a different failure mode — read the sibling skill `tinyworlds-grad-norm-diagnosis` for grad-norm interpretation, the Adam-preconditioner-decoupling mechanism, and the recovery playbook.
- It's not a checkpoint-loading artifact, although there *was* a coincident checkpoint corruption bug — see §2.

## 2. DDP checkpoint save-race bug + fix

### Symptom

In `results/2026_06_23_09_31_58_30k/dynamics/checkpoints/`, **28 of 30** `model_state_dict.pt` files were exactly **1,315 bytes** instead of the expected ~305 MB. The two intact ones (steps 0 and 17000) survived by chance.

1315 bytes is the on-disk size of `torch.save({})` — i.e. an empty Python dict.

### Root cause

In `utils/utils.py:save_training_state`:

```python
state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
optimizer_state_dict = get_optimizer_state_dict(...)
os.makedirs(ckpt_path, exist_ok=True)
torch.save(state_dict, Path(ckpt_path) / MODEL_CHECKPOINT)
```

Under DDP with `full_state_dict=True, cpu_offload=True`:

- `get_model_state_dict` is a **collective** that gathers parameters onto **rank 0 only**. Non-rank-0 ranks return `{}`.
- All 4 ranks then race to write the **same file path**.
- Non-rank-0 ranks (carrying `{}`) frequently win the race and clobber rank-0's valid checkpoint.

### Fix

`utils/utils.py:save_training_state` was patched to:

1. Still call `get_*_state_dict` on **all ranks** (it's a collective; skipping on non-rank-0 deadlocks).
2. Gate **file I/O** (`os.makedirs`, `torch.save`) to rank 0 only.
3. Add `dist.barrier()` after the writes so non-rank-0 workers don't continue while rank 0 is still flushing.

`scripts/train_dynamics.py` was patched analogously: the secondary `all_optimizers.pt` / `all_schedulers.pt` writes for split-optimizer (Muon+AdamW) configs are now gated `if len(optimizers) > 1 and is_main:`.

### Verification

- Standalone DDP smoke test: `scripts/diag_save_race_fix.py` runs a 4-rank toy training loop and asserts checkpoint integrity.
- Production verification: `results/2026_06_24_15_31_15` has **12 of 12** checkpoints at 305 MB after ~11k steps. Bug is closed.

> **Always run `diag_save_race_fix.py` after touching `save_training_state`** — the bug is silent in single-GPU runs and only manifests under DDP.

## 3. GH200 memory budget for dynamics training (4× 96 GiB DDP, AMP-bf16)

For the **18-block / 512-dim / 2048-hidden** dynamics model (the "large" config in `dynamics_large_lam.yaml`):

| bs/GPU | Result | Peak GPU memory |
|-------:|--------|----------------:|
| 32     | Fits   | ~50–53 GiB (production), ~70 GiB worst-case (documented) |
| 48     | Fits   | ~80 GiB (briefly tested) |
| 64     | OOM in backward | ~91 GiB |
| 128    | OOM in forward (first FiLM-modulated norm) | ~93 GiB |

The dominant memory cost is the FFN intermediate activations at `hidden_dim=2048` × 18 blocks.

### Launcher tunable

`scripts/launch_dynamics_4gpu.sh` exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to mitigate fragmentation under DDP+AMP — without it, the model OOMs at smaller batch sizes than the static budget would suggest.

### Override pattern

```bash
BATCH_SIZE_PER_GPU=32 GRAD_ACCUM_STEPS=4 USE_WANDB=true \
  bash scripts/launch_dynamics_4gpu.sh
```

Effective batch = `bs/GPU × grad_accum × nproc_per_node`. Target 512 to match the proven LR / β₂ recipe in `dynamics_large_lam.yaml`.

## 4. Params / tokens ratio for Zelda (the scaling-law lens)

### Zelda training-token count (with `training_large_lam.yaml` settings)

```
cached HDF5 (data/zelda_frames.h5):   72,410 frames
- load_start_index=1000:              71,410 frames
× preload_ratio=1.0:                  71,410 frames
× train_ratio=0.8:                    57,128 frames

× patches/frame = (64/4)² = 256:      14,624,768 tokens   (14.62 M)
```

Each "token" = 1 discrete FSQ index from a codebook of `num_bins^latent_dim = 4^5 = 1024`.

### Model-size comparison

| Config                          | Params | tokens/param | Chinchilla gap |
|---------------------------------|-------:|-------------:|---------------:|
| Large (current run): 18×512×2048 | ~76 M  | 0.19         | **105× under** |
| Medium (proposed): 12×256×1024  | 12.9 M | 1.13         | 17.7× under    |
| Chinchilla-optimal at this data | ~0.73 M | 20.0         | 1× (baseline)  |
| "Low overfit" target (~100 tok/param) | ~0.15 M | 100.0    | —              |

Computed from `models/dynamics.py` instantiation, not from YAML alone — `output_mlp` (vocab=1024) adds ~263k params on top of the transformer.

### Why this matters

The U-shape in §1 is unavoidable at any model size on this data while training to 30k updates. Even the proposed medium config still cycles each unique token through the loss ~800 times. **Reducing model size helps but does not fix the U** — it just shifts the eval-loss minimum to a later step and reduces its depth.

### What the LAM and VT contribute

- **VT** (`results/2026_06_08_12_51_05_zelda/video_tokenizer/checkpoints/video_tokenizer_step_37500`) and **LAM** (`results/large_action_model`, 67M params, n_actions=4) are **frozen** during dynamics training. No gradients, no optimizer state.
- Their parameter count does **not** affect the dynamics-stage compute/memory budget meaningfully.
- They define the **interface** the dynamics model consumes:
  - VT → token grid of shape `[B, T, P=256, L=5]` over vocab=1024.
  - LAM → action latent of `dim = log₂(n_actions) = 2`, projected to `embed_dim` by `AdaptiveNormalizer` (FiLM).
- Therefore: **resizing the dynamics model does not require changing VT or LAM.** Only the dynamics' `embed_dim`, `num_blocks`, `hidden_dim`, `num_heads` matter.

## 5. Inference-on-train-vs-eval — the right test and an inconclusive result

### Conceptual prediction

An overfit model should produce visibly better generation on **training clips** than on **eval clips**. The training objective is token-level CE — minimizing it implies the model assigns high probability to the exact training token sequences.

### Two lossy floors between training-CE and inference RGB

Even a model with near-zero training-token CE can have non-trivial inference RGB error because:

1. **VT lossy compression**: the frozen FSQ-VAE round-trip (token → RGB) has reconstruction error >0.
2. **MaskGIT iterative decoding** at inference is strictly harder than training: training fills masked patches *within* a fully-known sequence; inference must hallucinate the entire prediction horizon iteratively from only the context.

### What we tried (and why it didn't show the gap)

`scripts/compare_ckpt_inference.py` ran inference on **N=1 train clip + N=1 eval clip per checkpoint**:

```
ckpt    region   gen_only_mse
0       train      0.01747
0       eval       0.01370
17000   train      0.00857
17000   eval       0.00752
```

Both partitions improved at similar rates step-0 → step-17000, and **eval MSE is lower than train MSE at both checkpoints**. This does not show overfitting visually — but it doesn't disprove it either. N=1 per partition is too noisy: pixel MSE is dominated by background complexity, and the chosen eval clip happens to be visually simpler than the chosen train clip.

### The clean signal we already have

The **token-CE eval loss** in §1 is computed over the entire eval partition (~7k frames). The U-shape there is statistically reliable. The pixel-MSE comparison would need N≥10 clips per partition with averaged MSE/SSIM to show the same signal in pixel space.

> If you want a definitive visual answer: extend `compare_ckpt_inference.py` to sample ≥10 clips per partition, plot mean ± std MSE per (checkpoint, partition), and add side-by-side ground-truth-vs-generation PNG grids. The MSE distribution gap is the real evidence, not any single clip.

## 6. Recommended next experiments (cheapest → most expensive)

1. **Early stopping**: the current 30k run's best generalization checkpoint is around step_7000–8000. Just take that one. Don't run to 30k.
2. **Add dropout** to `models/st_transformer.py` (e.g. `p=0.1` after attention and FFN) and **bump `weight_decay` to 0.05**. Both cost zero compute, address the no-regularization problem head-on.
3. **Train the medium config** (12-block × 256 × 1024, ~12.9M params) as a control. Document whether the eval minimum is deeper (better) or just shifts later.
4. **Statistical inference comparison** (see §5) — turns the noisy single-clip MSE into a clean overfitting visualization for any pair of checkpoints.
5. **More data** is the only thing that fundamentally moves the params/tokens ratio. `preload_ratio` is already at 1.0; the next move is adding more raw Zelda footage to `data/zelda_frames.h5`.

## 7. Artifact locations

| Artifact | Location |
|---|---|
| Broken original run | `results/2026_06_23_09_31_58_30k/` (28/30 ckpts empty) |
| Clean re-run | `results/2026_06_24_15_31_15/` (12/12 ckpts intact at last check) |
| Inference comparison PNGs | `inference_results/ckpt_compare/` |
| HF mirror of pretrained components | https://huggingface.co/qtris123/tinyworlds-zelda (public) — VT @ step_37500, LAM, original dynamics checkpoints |
| Save-race fix smoke test | `scripts/diag_save_race_fix.py` |
| Custom inference comparator | `scripts/compare_ckpt_inference.py` |
| Launcher | `scripts/launch_dynamics_4gpu.sh` |
| Large-LAM dynamics config | `configs/dynamics_large_lam.yaml`, `configs/training_large_lam.yaml` |

## 8. Quick-reference numbers (cheat sheet)

```
Zelda training tokens (unique):    14.62 M
Zelda eval tokens (unique):         1.83 M
Patches per 64×64 frame:              256
FSQ vocab:                          1024  (4^5)
Context length:                        4 frames
frame_skip (15 fps):                   4
Action dim (n_actions=4):              2

Large dynamics params:               ~76 M
Medium dynamics params (proposed):  12.9 M

Effective batch (bs=32 × ga=4 × 4 GPUs):  512
Target tokens per step (~75% mask):      ~393 k
Total target-token presentations / 30k:  ~11.8 B
Each unique token presented:            ~800 ×

Eval-loss minimum (current run):    step ~7000–8000
Per-checkpoint size:                 305 MB
1315-byte "empty dict" bug signature: torch.save({})
```
