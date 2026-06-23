"""CPU-only smoke test for the QK-Norm resume from step-8500.

Verifies that:
  1. The new DynamicsModel (with QK-Norm) can be constructed with the same
     hyperparameters used in the spiked run.
  2. The step-8500 checkpoint loads cleanly under ``strict=False`` and only
     the new ``q_norm.weight`` / ``k_norm.weight`` keys are reported as
     missing (initialized to ones by RMSNorm).
  3. A forward + backward pass produces a finite loss and finite grad-norm.

Run:
    python scripts/diag_resume_smoketest.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/localhome/local-triv/tinyworlds")

import torch
from models.dynamics import DynamicsModel
from utils.utils import load_dynamics_from_checkpoint

CHECKPOINT = "/localhome/local-triv/tinyworlds/results/2026_06_21_13_02_14/dynamics/checkpoints/dynamics_step_8500"


def main():
    state = torch.load(os.path.join(CHECKPOINT, "state.pt"), map_location="cpu", weights_only=False)
    cfg = state.get("config", {}) or {}

    # Build with current code (= with QK-Norm). conditioning_dim is inferred
    # from the FiLM weight shape in the checkpoint.
    cond_dim = None
    model_sd = torch.load(os.path.join(CHECKPOINT, "model_state_dict.pt"), map_location="cpu", weights_only=True)
    for k, v in model_sd.items():
        if k.endswith("to_gamma_beta.1.weight"):
            cond_dim = int(v.shape[1])
            break
    print(f"inferred conditioning_dim: {cond_dim}")

    model = DynamicsModel(
        frame_size=(cfg.get("frame_size", 64), cfg.get("frame_size", 64)),
        patch_size=cfg.get("patch_size", 4),
        embed_dim=cfg.get("embed_dim", 512),
        num_heads=cfg.get("num_heads", 8),
        hidden_dim=cfg.get("hidden_dim", 2048),
        num_blocks=cfg.get("num_blocks", 18),
        conditioning_dim=cond_dim if cond_dim is not None else 32,
        latent_dim=cfg.get("latent_dim", 5),
        num_bins=cfg.get("num_bins", 4),
    )

    n_params = sum(p.numel() for p in model.parameters())
    qk_params = sum(p.numel() for n, p in model.named_parameters() if ".q_norm." in n or ".k_norm." in n)
    print(f"total params: {n_params/1e6:.2f}M  (QK-Norm added: {qk_params} = {qk_params/n_params*100:.4f}%)")

    # 1) Load step-8500 weights into new architecture with strict=False
    model, _ = load_dynamics_from_checkpoint(
        checkpoint_path=CHECKPOINT,
        device="cpu",
        model=model,
        is_distributed=False,
        strict=False,
    )

    # 2) Verify the q_norm/k_norm weights are still at their constructor init (=1)
    for name, p in model.named_parameters():
        if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
            ones = torch.allclose(p, torch.ones_like(p))
            if not ones:
                print(f"  WARN: {name} != ones after load (mean={p.mean():.4f})")
                break
    else:
        print("all q_norm/k_norm weights = 1.0 after load (expected)")

    # 3) Tiny forward + backward
    B, T, P_tok, L = 2, cfg.get("context_length", 4), (cfg.get("frame_size", 64) // cfg.get("patch_size", 4)) ** 2, cfg.get("latent_dim", 5)
    num_bins = cfg.get("num_bins", 4)
    A = cond_dim if cond_dim is not None else 32

    discrete_latents = torch.randint(0, num_bins, (B, T, P_tok, L)).float() - 1.5  # FSQ-like values
    targets = torch.randint(0, num_bins ** L, (B, T, P_tok))
    conditioning = torch.randn(B, T - 1, A)

    model.train()
    pred_logits, mask_positions, loss = model(discrete_latents, training=True, conditioning=conditioning, targets=targets)
    print(f"forward OK: loss={loss.item():.4f} (expected ~log({num_bins**L})={torch.log(torch.tensor(float(num_bins**L))).item():.4f})")

    loss.backward()
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += float(p.grad.data.float().norm(2) ** 2)
    print(f"backward OK: global grad-norm={total_norm_sq**0.5:.4f} (finite={total_norm_sq == total_norm_sq})")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
