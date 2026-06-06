# Mamba-3 block swap — implementation reference

Detailed plan for replacing STTransformer attention with a Mamba-3 SSM time-mixer in TinyWorlds. Read `SKILL.md` first for the architecture/contract overview.

## Where the change lands

All edits are in `models/st_transformer.py` plus config plumbing. The block to modify:

```201:223:models/st_transformer.py
class STTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, causal=True, conditioning_dim=None,
                 use_moe=False, num_experts=4, top_k_experts=2, moe_aux_loss_coeff=0.01):
        super().__init__()
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, conditioning_dim)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, causal, conditioning_dim)
        ...
    def forward(self, x, conditioning=None):
        x = self.spatial_attn(x, conditioning)
        x = self.temporal_attn(x, conditioning)
        x = self.ffn(x, conditioning)
        return x
```

Strategy: introduce a `MambaTemporalMixer` with the **same interface** as `TemporalAttention.forward(x, conditioning) -> x` and select it via a `temporal_mixer` flag. This preserves the spatial-attn + temporal-mix + FFN structure, the residual stream, and FiLM conditioning.

## Interface contract the new mixer MUST honor

- Signature: `forward(self, x, conditioning=None)`; `x: [B, T, P, E]` in, `[B, T, P, E]` out.
- Apply residual then `AdaptiveNormalizer(embed_dim, conditioning_dim)` (reuse `self.norm` exactly like `TemporalAttention`) so action FiLM still modulates the output. The `AdaptiveNormalizer` already handles `T-1` conditioning left-padding.
- Causality: an SSM recurrence is causal by construction over the scanned axis. Scan over `T`, never over `P`.
- Reshape so the SSM sees time as the sequence axis, batching over patches:
  `rearrange(x, 'b t p e -> (b p) t e')` → SSM → `rearrange(out, '(b p) t e -> b t p e', b=B)`.

## Dependency options (pick one, document it)

1. **Official kernels**: `pip install mamba-ssm` (and `causal-conv1d`). CUDA-only, fastest. Reference `Mamba3`/`Mamba2` block from the release. Pin versions in `requirements.txt`.
2. **Pure-PyTorch fallback**: a minimal selective-scan in plain PyTorch (slow but device-agnostic, works on `mps`/CPU for dev). Use for correctness checks and small `configs/dev/` runs.

Provide both behind the config flag so contributors without CUDA can still run dev configs.

## Skeleton (pure-PyTorch fallback, SISO; add complex/RoPE + MIMO incrementally)

```python
# models/mamba3.py
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from models.norms import AdaptiveNormalizer

class MambaTemporalMixer(nn.Module):
    """Drop-in replacement for TemporalAttention over the T axis. [B,T,P,E] -> [B,T,P,E]."""
    def __init__(self, embed_dim, d_state=16, d_conv=4, expand=2, conditioning_dim=None):
        super().__init__()
        d_inner = expand * embed_dim
        self.in_proj  = nn.Linear(embed_dim, 2 * d_inner)         # x and gate z
        self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1)        # B, C, dt (data-dependent)
        self.dt_proj  = nn.Linear(1, d_inner)
        self.A_log    = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1))
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, embed_dim)
        self.norm     = AdaptiveNormalizer(embed_dim, conditioning_dim)  # keep FiLM contract

    def _scan(self, u):                                            # u: [N, T, d_inner]
        N, T, Di = u.shape
        xz = self.in_proj(u)                                       # via caller; placeholder
        # ... selective state-space recurrence over T (causal), with
        #     dt = softplus(dt_proj(...)), A = -exp(A_log), discretize, accumulate state ...
        return u                                                   # replace with real scan output

    def forward(self, x, conditioning=None):
        B, T, P, E = x.shape
        u = rearrange(x, 'b t p e -> (b p) t e')
        # in_proj -> split (x, z); causal conv1d over T (slice [..., :T]); selective scan;
        # gate by SiLU(z); out_proj. (Fill in per Mamba-3 recurrence.)
        y = self._scan(u)                                          # [(B P), T, E]
        y = rearrange(y, '(b p) t e -> b t p e', b=B)
        return self.norm(x + y, conditioning)                     # residual + FiLM/RMS norm
```

Then wire selection in `STTransformerBlock.__init__`:

```python
if temporal_mixer == "mamba3":
    self.temporal_mix = MambaTemporalMixer(embed_dim, conditioning_dim=conditioning_dim)
else:
    self.temporal_mix = TemporalAttention(embed_dim, num_heads, causal, conditioning_dim)
# forward(): x = self.spatial_attn(x, c); x = self.temporal_mix(x, c); x = self.ffn(x, c)
```

Mamba-3 specifics to add on top of the SISO skeleton (per the paper):
- **Exponential-trapezoidal discretization**: second-order recurrence instead of first-order Euler `dt*A`.
- **Complex state / data-dependent RoPE**: split `A` into real `A` and imaginary `Theta`; apply RoPE-style rotation to B/C — when enabled, drop the additive temporal sin-cos PE for these blocks.
- **BC/QK-norm**: normalize B and C projections (stabilizes training).
- **MIMO (optional)**: expand B/C projections for multi-input/output; bigger accuracy, same decode latency, longer training.

## Config plumbing

Add `temporal_mixer: "attention"` (default) to `configs/training.yaml` and each stage YAML, plus optional `mamba_d_state`, `mamba_d_conv`, `mamba_expand`. Add the fields to the dataclasses in `utils/config.py` (`TrainingConfig`, `VideoTokenizerConfig`, `LatentActionsConfig`, `DynamicsConfig`) and pass them through:
- `STTransformer(...)` constructor → `STTransformerBlock(...)`.
- Each model that builds an `STTransformer`: `models/video_tokenizer.py` (encoder + decoder), `models/latent_actions.py` (encoder + decoder), `models/dynamics.py`.
- Each `models/*.py` constructor is called from its `scripts/train_*.py`; thread the new args there too.

Keep `STTransformer.moe_aux_loss()` / `moe_expert_utilization()` intact (the FFN is unchanged).

## Recommended rollout order

1. Pure-PyTorch SISO mixer, temporal-only, on `configs/dev/dev_training.yaml` — verify shapes, loss decreases, parity vs attention on a tiny run.
2. Swap into the **dynamics** stage first (largest model, `num_blocks=8`, most compute) — that's where linear-time scan pays off and where quality matters most for rollouts.
3. Add complex/RoPE state update, then BC-norm; disable additive temporal PE for SSM blocks.
4. Optionally swap the tokenizer/LAM temporal mixers too; optionally add MIMO.
5. For speed, switch to `mamba-ssm` CUDA kernels behind the same flag.

## Ablation / evaluation protocol

Train attention baseline and Mamba-3 variant with identical configs (same `embed_dim`, `num_blocks`, `n_updates`, data, seed). Compare:
- Stage losses: tokenizer/LAM `smooth_l1` recon, dynamics masked cross-entropy (from W&B `train/loss`).
- Param count (`print_param_count_if_main`) — confirm parameter-matched.
- Throughput / step time and peak memory, especially at longer `context_length` (raise it to stress the quadratic-vs-linear gap).
- Rollout quality from `scripts/run_inference.py`: visual coherence over `generation_steps`, action controllability (interactive vs `use_gt_actions`).
- Sweep `context_length` up (e.g. 4 → 8 → 16) to show the SSM's scaling advantage that the attention baseline cannot match within memory.
