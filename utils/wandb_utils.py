import wandb
import torch
import os
import math
import time
from typing import Any, Dict, Iterable, Optional, Tuple


def init_wandb(project_name: str, config: Dict[str, Any], run_name: Optional[str] = None) -> wandb.run:

    # Generate run name if not provided
    if run_name is None:
        run_name = f"{project_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize wandb
    run = wandb.init(
        project=project_name,
        config=config,
        name=run_name,
        tags=[project_name, "training"]
    )
    
    print(f"🚀 W&B run initialized: {run.name}")
    print(f"📊 Project: {project_name}")
    print(f"🔗 View at: {run.url}")
    
    return run


def log_training_metrics(step: int, metrics: Dict[str, float], prefix: str = "train"):
    # Add prefix to metric names
    prefixed_metrics = {f"{prefix}/{k}": (v.item() if hasattr(v, "item") else float(v)) for k, v in metrics.items()}
    wandb.log(prefixed_metrics, step=step)


def log_learning_rate(optimizer: torch.optim.Optimizer, step: int):
    for i, param_group in enumerate(optimizer.param_groups):
        wandb.log({
            f"learning_rate/group_{i}": float(param_group['lr']),
        }, step=step)

def log_codebook_usage(codebook_usage: float, step: int, model_name: str = "model"):
    wandb.log({
        f"{model_name}/codebook_usage": float(codebook_usage),
    }, step=step)


def log_action_distribution(action_indices: torch.Tensor, step: int, n_actions: int):
    # Convert to CPU and numpy
    flat = action_indices.detach().cpu().reshape(-1)
    # Compute distribution
    counts = torch.bincount(flat.long(), minlength=n_actions).float()
    probs = counts / counts.sum().clamp_min(1)
    
    wandb.log({
        "action_distribution": wandb.Histogram(flat.numpy()),
        "action_entropy": float(-(probs * (probs + 1e-8).log()).sum()),
        "unique_actions": int((counts > 0).sum().item()),
    }, step=step)


def clip_and_log_grad_norm(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float = 1.0,
    *,
    step: int,
    log_to_wandb: bool = True,
    prefix: str = "train",
) -> Tuple[float, bool]:
    """Clip gradients to ``max_norm`` and report the pre-clip global L2 norm.

    Returns ``(pre_clip_norm, nonfinite)``. The caller decides whether to skip
    the optimizer step when ``nonfinite`` is True (recommended: skip and zero
    the grads so a bad batch cannot poison AdamW's moment estimates).

    When ``log_to_wandb`` is True and a wandb run is active, logs:
      - ``{prefix}/grad_norm``    — pre-clip global L2 norm
      - ``{prefix}/grad_clipped`` — 1.0 if the norm exceeded ``max_norm`` else 0.0
      - ``{prefix}/grad_skipped`` — 1.0 if the norm was NaN/Inf else 0.0
    """
    pre_clip = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    pre_clip_val = float(pre_clip.detach().cpu()) if torch.is_tensor(pre_clip) else float(pre_clip)
    nonfinite = not math.isfinite(pre_clip_val)
    if log_to_wandb and wandb.run is not None:
        wandb.log({
            f"{prefix}/grad_norm": pre_clip_val,
            f"{prefix}/grad_clipped": float((not nonfinite) and pre_clip_val > max_norm),
            f"{prefix}/grad_skipped": float(nonfinite),
        }, step=step)
    return pre_clip_val, nonfinite


def log_system_metrics(step: int):
    if torch.cuda.is_available():
        wandb.log({
            "system/gpu_memory_allocated": float(torch.cuda.memory_allocated() / 1024**3),  # GB
            "system/gpu_memory_reserved": float(torch.cuda.memory_reserved() / 1024**3),    # GB
        }, step=step)


def log_data_partition(partition_info: Dict[str, Any]) -> None:
    """Log dataset partition metadata to the active wandb run's config.
    Shows up in the wandb Config panel (not as a metric) since it is
    static metadata that describes the dataset, not a time-series value.
    """
    if wandb.run is None:
        return
    wandb.config.update({"data": partition_info}, allow_val_change=True)


def finish_wandb():
    """Finish the W&B run"""
    if wandb.run is not None:
        wandb.finish()


def create_wandb_config(args, model_config: Dict[str, Any]) -> Dict[str, Any]:
    config = {
        # Training parameters
        "batch_size": args.batch_size,
        "n_updates": args.n_updates,
        "learning_rate": args.learning_rate,
        "log_interval": getattr(args, 'log_interval', 100),
        
        # Dataset parameters
        "dataset": getattr(args, 'dataset', 'SONIC'),
        "context_length": getattr(args, 'context_length', 4),
        
        # Model architecture
        "model_architecture": model_config,
        
        # System parameters
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    # Add any additional args that might exist
    for attr in dir(args):
        if not attr.startswith('_') and not callable(getattr(args, attr)):
            if attr not in config:
                config[attr] = getattr(args, attr)
    
    return config
