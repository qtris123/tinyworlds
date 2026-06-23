"""CPU-only diagnostic for the step-8.8k loss spike.

Compares checkpoints at steps 8500 (last healthy), 9000 (first post-spike) and 9500
(plateau) to determine what blew up. We look for:
  * NaN / Inf in parameters
  * Per-tensor stats (mean abs, max abs, std) — to find the layer that exploded
  * AdamW optimizer state: max/min of `exp_avg_sq` (v) — small v with a big grad
    leads to giant updates; NaN in v poisons the parameter forever.

Run from /localhome/local-triv/tinyworlds with the `tiny` env active:
    python scripts/diag_loss_spike.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch

RUN_DIR = Path("/localhome/local-triv/tinyworlds/results/2026_06_21_13_02_14/dynamics/checkpoints")
STEPS = [8500, 9000, 9500]


def tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().float()
    nan = int(torch.isnan(t).sum().item())
    inf = int(torch.isinf(t).sum().item())
    if t.numel() == 0:
        return {"nan": nan, "inf": inf, "mean_abs": 0.0, "max_abs": 0.0, "std": 0.0}
    finite_mask = torch.isfinite(t)
    t_finite = t[finite_mask]
    if t_finite.numel() == 0:
        return {"nan": nan, "inf": inf, "mean_abs": float("nan"), "max_abs": float("nan"), "std": float("nan")}
    return {
        "nan": nan,
        "inf": inf,
        "mean_abs": float(t_finite.abs().mean()),
        "max_abs": float(t_finite.abs().max()),
        "std": float(t_finite.std()),
    }


def load_model_sd(step: int) -> Dict[str, torch.Tensor]:
    p = RUN_DIR / f"dynamics_step_{step}" / "model_state_dict.pt"
    return torch.load(p, map_location="cpu", weights_only=True)


def load_optim_sd(step: int) -> Dict:
    p = RUN_DIR / f"dynamics_step_{step}" / "optim_state_dict.pt"
    return torch.load(p, map_location="cpu", weights_only=False)


def summarize_model(step: int) -> Dict[str, Dict[str, float]]:
    sd = load_model_sd(step)
    return {k: tensor_stats(v) for k, v in sd.items()}


def summarize_optim(step: int) -> Dict[str, Dict[str, float]]:
    osd = load_optim_sd(step)
    # AdamW fused state_dict has top-level 'state' keyed by integer param index,
    # each containing {'step','exp_avg','exp_avg_sq'}.
    out = {}
    state = osd.get("state", osd)
    for pid, ps in state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in ps and isinstance(ps[key], torch.Tensor):
                out[f"param{pid}/{key}"] = tensor_stats(ps[key])
    return out


def print_globals(label: str, stats: Dict[str, Dict[str, float]]):
    total_nan = sum(s["nan"] for s in stats.values())
    total_inf = sum(s["inf"] for s in stats.values())
    max_abs = max((s["max_abs"] for s in stats.values() if s["max_abs"] == s["max_abs"]), default=0.0)
    print(f"  [{label}] tensors={len(stats):4d}  total_NaN={total_nan:>10d}  total_Inf={total_inf:>10d}  global_max_abs={max_abs:.3e}")


def top_movers(prev: Dict[str, Dict[str, float]], curr: Dict[str, Dict[str, float]], k: int = 10, key: str = "max_abs"):
    """Print top-k tensors whose `key` stat grew most between checkpoints."""
    rows = []
    for name in curr:
        if name not in prev:
            continue
        p, c = prev[name][key], curr[name][key]
        if p == 0 or p != p or c != c:  # NaN-safe
            ratio = float("inf") if c > 0 else 0.0
        else:
            ratio = c / p
        rows.append((name, p, c, ratio, curr[name]["nan"], curr[name]["inf"]))
    rows.sort(key=lambda r: (r[3] if r[3] == r[3] else 0.0), reverse=True)
    for name, p, c, ratio, nan, inf in rows[:k]:
        print(f"    {name:60s}  {p:9.3e} -> {c:9.3e}  x{ratio:8.2f}  nan={nan} inf={inf}")


def find_corrupted(stats: Dict[str, Dict[str, float]], k: int = 10):
    rows = [(name, s) for name, s in stats.items() if s["nan"] > 0 or s["inf"] > 0]
    rows.sort(key=lambda r: r[1]["nan"] + r[1]["inf"], reverse=True)
    for name, s in rows[:k]:
        print(f"    CORRUPTED  {name:60s}  nan={s['nan']:>8d} inf={s['inf']:>8d} max_abs={s['max_abs']:.3e}")


def main():
    print("=" * 100)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 100)
    model_stats = {}
    for step in STEPS:
        print(f"\n>>> step {step}")
        model_stats[step] = summarize_model(step)
        print_globals(f"step {step}", model_stats[step])
        find_corrupted(model_stats[step])

    print("\n" + "=" * 100)
    print("MODEL: top movers (max_abs) per transition")
    print("=" * 100)
    for a, b in zip(STEPS, STEPS[1:]):
        print(f"\n[{a} -> {b}]  top-10 by max_abs growth ratio:")
        top_movers(model_stats[a], model_stats[b], k=12, key="max_abs")

    print("\n" + "=" * 100)
    print("OPTIMIZER STATE SUMMARY (AdamW exp_avg / exp_avg_sq)")
    print("=" * 100)
    optim_stats = {}
    for step in STEPS:
        print(f"\n>>> step {step}")
        optim_stats[step] = summarize_optim(step)
        print_globals(f"opt step {step}", optim_stats[step])
        find_corrupted(optim_stats[step])

    print("\n" + "=" * 100)
    print("OPTIMIZER: top movers per transition")
    print("=" * 100)
    for a, b in zip(STEPS, STEPS[1:]):
        print(f"\n[{a} -> {b}]  top-10 by max_abs growth ratio:")
        top_movers(optim_stats[a], optim_stats[b], k=12, key="max_abs")

    print("\nDone.")


if __name__ == "__main__":
    main()
