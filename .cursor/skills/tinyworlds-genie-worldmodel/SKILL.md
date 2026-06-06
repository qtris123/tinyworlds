---
name: tinyworlds-genie-worldmodel
description: Architecture, training, and inference reference for the TinyWorlds Genie-style world model (FSQ-VAE video tokenizer + latent action model + MaskGIT dynamics over a Space-Time Transformer). Use when reproducing TinyWorlds paper/repo results, editing models/ or configs/, running scripts/full_train.py or run_inference.py, or swapping STTransformer attention blocks for a Mamba-3 SSM block.
---

# TinyWorlds (Genie-style World Model)

TinyWorlds is a minimal reimplementation of DeepMind's Genie ([arXiv:2402.15391](https://arxiv.org/pdf/2402.15391)): an autoregressive, action-controllable video world model trained on unlabeled gameplay video. All paths below are relative to the repo root (`tinyworlds/`).

## Three-stage pipeline

Trained sequentially; each later stage consumes the frozen output of earlier stages.

| Stage | Module (`models/`) | Trains | Loss |
|-------|-------------------|--------|------|
| 1. Video Tokenizer | `video_tokenizer.py` | FSQ-VAE: frames → discrete tokens → frames | `smooth_l1` reconstruction |
| 2. Latent Action Model (LAM) | `latent_actions.py` | FSQ-VAE: infers a discrete action per frame transition | `smooth_l1` recon + variance penalty (`var_lambda=100`, `var_target=0.01`) |
| 3. Dynamics | `dynamics.py` | MaskGIT predictor of next video tokens given past tokens + action | masked cross-entropy (+ optional MoE aux loss) |

Inference loop (`scripts/run_inference.py`): tokenize context frames → pick an action token → dynamics iteratively unmasks next-frame tokens → tokenizer decodes to pixels → repeat.

## Shared backbone: Space-Time Transformer (`models/st_transformer.py`)

Tensors flow as `[B, T, P, E]` (batch, time, patches-per-frame, embed). `STTransformerBlock` applies three sublayers, each a residual + normalize:

1. `SpatialAttention` — full attention across `P` patches within a frame.
2. `TemporalAttention` — attention across `T` (causal when `causal=True`), per patch position.
3. FFN — `SwiGLUFFN` by default, or `MoESwiGLUFFN` (top-k routed experts, dynamics only when `use_moe=true`).

Key details that matter for any block swap:
- **Normalization is fused into each sublayer** via `AdaptiveNormalizer` (`models/norms.py`): unconditioned → `RMSNorm`; conditioned (`conditioning_dim` set) → FiLM, i.e. `SimpleLayerNorm(x) * (1 + gamma) + beta` where `gamma, beta = MLP(conditioning)`. Conditioning is the action embedding for LAM decoder and dynamics.
- **Action conditioning timing**: when `conditioning` has `T-1` steps, `AdaptiveNormalizer` left-pads with zeros so action `a_{t-1}` modulates frame `z_t`.
- **Positional encoding** is added inside `STTransformer.forward` and by callers: spatial 2D sin-cos PE fills the first ~2/3 of `E`; temporal sin-cos PE (`sincos_time`) fills the last 1/3 (`temporal_dim = (embed_dim // 3) & ~1`). There is **no RoPE** (listed as a TODO in the README).
- `STTransformer.moe_aux_loss()` / `moe_expert_utilization()` aggregate per-block MoE stats.

`embed_dim` must be divisible by `num_heads`. FSQ codebook size is `num_bins ** latent_dim` (default `4**5 = 1024`). `n_actions` must be a power of 2 (`action_dim = log2(n_actions)`, fixed `NUM_LATENT_ACTIONS_BINS = 2`).

## Config system (`configs/`, `utils/config.py`)

`configs/training.yaml` holds shared params and is overlaid onto each per-stage YAML (training values win; CLI dotlist `key=value` overrides win over everything). Loaders: `load_config` and `load_stage_config_merged`. Configs are typed dataclasses (`VideoTokenizerConfig`, `LatentActionsConfig`, `DynamicsConfig`, `TrainingConfig`, `InferenceConfig`).

Shared defaults (`configs/training.yaml`): `patch_size=4`, `context_length=4`, `frame_size=64`, `latent_dim=5`, `num_bins=4`, `n_actions=4`, `embed_dim=32`, `num_heads=8`, `hidden_dim=128`, `optimizer=adamw` (or `muon`), `use_moe=false`, `amp/tf32/compile=true`, `dataset=PICODOOM`.

Per-stage training defaults:

| Stage | config | batch | n_updates | lr | num_blocks | log_interval |
|-------|--------|-------|-----------|----|-----------|--------------|
| Tokenizer | `video_tokenizer.yaml` | 350 | 40,000 | 1e-3 | 4 | 2500 |
| LAM | `latent_actions.yaml` | 350 | 10,000 | 1e-4 | 2 | 500 |
| Dynamics | `dynamics.yaml` | 500 | 300,000 | 1e-2 | 8 | 2000 |

Inference (`configs/inference.yaml`): `generation_steps=10`, `context_window=2`, `prediction_horizon=1`, `temperature=0.5` (0=argmax), `device=mps`, `use_interactive_mode=true`. Dynamics MaskGIT decode uses an exponential unmask schedule (`schedule_k=5.0`).

> Dynamics model params (`patch_size`, `embed_dim`, `latent_dim`, `num_bins`, `frame_size`) MUST match the trained tokenizer, or token indices won't align.

## Reproducing results

```bash
pip install -r requirements.txt          # torch>=2.8, einops, h5py, omegaconf, wandb, ...
python scripts/download_assets.py         # datasets -> data/, pretrained -> results/
python scripts/full_train.py --config configs/training.yaml   # runs all 3 stages in order
python scripts/run_inference.py --config configs/inference.yaml
```

- `scripts/full_train.py` orchestrates the 3 stages as subprocesses, auto-discovering the latest tokenizer/LAM checkpoints and passing them into dynamics via `video_tokenizer_path=`/`latent_actions_path=` overrides.
- Run a single stage directly, e.g. `python scripts/train_dynamics.py --config configs/dynamics.yaml --training_config configs/training.yaml video_tokenizer_path=<dir> latent_actions_path=<dir>`.
- Checkpoints land in `results/<timestamp>/<stage>/`. Dynamics loads tokenizer + LAM frozen (`requires_grad=False`, `.eval()`).
- Datasets (`datasets/datasets.py`): `PICODOOM`, `PONG`, `ZELDA`, `POLEPOSITION`, `SONIC`; `.mp4` → resized 64×64 RGB → `.h5`, normalized to `[-1, 1]`. Use `configs/dev/dev_training.yaml` for a fast smoke test.
- Acceleration: AMP (bf16), TF32, `torch.compile`, and DDP/FSDP (`utils/distributed.py`, requires `device=cuda`; FSDP is incompatible with `amp=true`). Optimizers: AdamW or Muon split (`models/muon.py`, `utils/optimizer_utils.py`) with cosine schedule.

## Improving the paper: swap attention blocks for a Mamba-3 block

Goal: replace the quadratic `TemporalAttention` (and optionally `SpatialAttention`) inside `STTransformerBlock` with a **Mamba-3** SSM block for linear-time, constant-memory sequence modeling, then compare against the attention baseline.

Mamba-3 ([arXiv:2603.15569](https://arxiv.org/html/2603.15569), Tri Dao 2026) upgrades Mamba-2 with: exponential-trapezoidal discretization (more expressive recurrence), complex-valued state updates via data-dependent RoPE (state tracking), an optional MIMO variant, and QK/BC-norm. It keeps the Mamba-2 block layout and is parameter-matched.

**Recommended scope**: swap **temporal** mixing only (the causal sequence-over-time dimension is the natural SSM fit); keep `SpatialAttention` as full attention over patches within a frame (no causal order, small `P`). This is the highest-leverage, lowest-risk change.

Integration constraints (read before coding):
- The new block must accept and return `[B, T, P, E]` and preserve the residual + `AdaptiveNormalizer(conditioning)` contract so FiLM action conditioning still works.
- A temporal SSM scans over `T` per patch: reshape `[B, T, P, E] -> [(B P), T, E]`, scan, reshape back. Honor causality (it's inherent to the recurrence) to match `causal=True`.
- The temporal sin-cos PE added in `STTransformer.forward` becomes redundant for an SSM time-mixer; Mamba-3's complex/RoPE update already encodes position. Consider gating it off for SSM blocks.
- Keep `STTransformer`'s `moe_aux_loss()` / `moe_expert_utilization()` working (don't remove the FFN).
- Add a config flag (e.g. `temporal_mixer: "attention" | "mamba3"`) threaded through `TrainingConfig`/stage configs and the model constructors so you can A/B test without code edits.

For the concrete drop-in plan, code skeleton, dependency options, and ablation/eval protocol, read [reference.md](reference.md).
