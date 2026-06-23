"""Map AdamW optimizer param indices to model parameter names.

The optimizer was built with two param groups (decay + no_decay), so the
optimizer-state index N corresponds to the N-th entry in
   list(group_decay['params']) + list(group_no_decay['params']).

We rebuild the same grouping by name and print the mapping for the indices
that exploded in the diagnostic.
"""
from __future__ import annotations

import sys
import os
import torch

# Need to import via the repo's package layout
sys.path.insert(0, "/localhome/local-triv/tinyworlds")

from models.dynamics import DynamicsModel

CHECKPOINT = "/localhome/local-triv/tinyworlds/results/2026_06_21_13_02_14/dynamics/checkpoints/dynamics_step_8500"

INTERESTING = [197, 198, 199, 200, 43, 138, 139, 171, 193, 194,
               376, 382, 410, 412, 430, 463, 464, 495, 501, 502, 506, 507]


def main():
    state = torch.load(os.path.join(CHECKPOINT, "state.pt"), map_location="cpu", weights_only=False)
    cfg = state.get("config", {})

    model = DynamicsModel(
        frame_size=(cfg.get("frame_size", 64), cfg.get("frame_size", 64)),
        patch_size=cfg.get("patch_size", 4),
        embed_dim=cfg.get("embed_dim", 512),
        num_heads=cfg.get("num_heads", 8),
        hidden_dim=cfg.get("hidden_dim", 2048),
        num_blocks=cfg.get("num_blocks", 18),
        conditioning_dim=cfg.get("conditioning_dim", 32),
        latent_dim=cfg.get("latent_dim", 5),
        num_bins=cfg.get("num_bins", 4),
    )

    # Same split as utils/optimizer_utils.py::_split_decay_params
    decay, no_decay = [], []
    decay_names, no_decay_names = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
            no_decay.append(param)
            no_decay_names.append((name, tuple(param.shape)))
        else:
            decay.append(param)
            decay_names.append((name, tuple(param.shape)))

    flat = decay_names + no_decay_names  # AdamW param_groups list order
    print(f"Total trainable params: {len(flat)}  (decay={len(decay_names)}, no_decay={len(no_decay_names)})")
    print()
    print(f"{'idx':>4}  {'name':70s}  {'shape'}")
    for idx in sorted(set(INTERESTING)):
        if idx < len(flat):
            name, shape = flat[idx]
            print(f"{idx:>4}  {name:70s}  {shape}")
        else:
            print(f"{idx:>4}  <out of range>")


if __name__ == "__main__":
    main()
