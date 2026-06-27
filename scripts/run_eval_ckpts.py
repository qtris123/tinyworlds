"""Run inference for a list of dynamics checkpoints on the same dataset sample(s).

This loads the video tokenizer, latent action model and the dataset only once,
then iterates over the requested dynamics checkpoints, generating a side-by-side
PNG and an mp4 for each. Outputs are written to
``inference_results/<run_label>/dynamics_step_<N>/``.

Usage:
    python scripts/run_eval_ckpts.py \
        --vt results/.../video_tokenizer/checkpoints/video_tokenizer_step_37500 \
        --lam results/.../latent_actions/checkpoints/latent_actions_step_9500 \
        --dyn-root results/dynamics/checkpoints \
        --steps 2000 10000 18000 24000 \
        --dataset ZELDA \
        --seed 42
"""

import argparse
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import torch

from datasets.data_utils import load_data_and_data_loaders
from utils.utils import (
    load_videotokenizer_from_checkpoint,
    load_latent_actions_from_checkpoint,
    load_dynamics_from_checkpoint,
)
from utils.inference_utils import (
    visualize_inference,
    get_action_latent,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vt', required=True, help='Video tokenizer checkpoint directory')
    p.add_argument('--lam', required=True, help='Latent actions checkpoint directory')
    p.add_argument('--dyn-root', required=True, help='Directory holding dynamics_step_* subdirs')
    p.add_argument('--steps', type=int, nargs='+', required=True, help='Steps to evaluate')
    p.add_argument('--dataset', default='ZELDA')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--context-window', type=int, default=4)
    p.add_argument('--generation-steps', type=int, default=8)
    p.add_argument('--prediction-horizon', type=int, default=1)
    p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--fps', type=int, default=2)
    p.add_argument('--teacher-forced', action='store_true', default=True)
    p.add_argument('--no-teacher-forced', dest='teacher_forced', action='store_false')
    p.add_argument('--preload-ratio', type=float, default=0.2,
                   help='Fraction of cached frames to load (lower = less RAM, fewer choices)')
    p.add_argument('--use-gt-actions', action='store_true', default=True,
                   help='Use latent-action-model-inferred actions for context')
    p.add_argument('--use-actions', action='store_true', default=False,
                   help='Sample random actions instead of GT actions')
    p.add_argument('--no-actions', action='store_true', default=False,
                   help='Disable action conditioning entirely')
    p.add_argument('--out-root', default='inference_results',
                   help='Root directory for generated outputs')
    p.add_argument('--run-label', default=None,
                   help='Optional subdir under --out-root (default: ckpts_<timestamp>)')
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_checkpoint(
    dyn_ckpt: str,
    args,
    video_tokenizer,
    latent_action_model,
    ground_truth_frames: torch.Tensor,
    use_actions: bool,
    out_dir: Path,
):
    """Run autoregressive inference for a single dynamics checkpoint.

    The action policy is intentionally re-seeded per checkpoint so the same
    sequence of (frame, action) pairs is presented to each model.
    """
    set_seed(args.seed)  # re-seed before random-action sampling

    dynamics_model, _ = load_dynamics_from_checkpoint(dyn_ckpt, args.device)
    dynamics_model.eval()

    cfg = SimpleNamespace(
        device=args.device,
        context_window=args.context_window,
        prediction_horizon=args.prediction_horizon,
        temperature=args.temperature,
        use_interactive_mode=False,
        use_gt_actions=args.use_gt_actions and not args.no_actions,
        use_actions=args.use_actions and not args.no_actions,
    )

    n_actions = latent_action_model.quantizer.codebook_size if use_actions else None
    inferred_actions: list = []

    effective_steps = args.generation_steps
    if args.teacher_forced:
        max_possible_steps = ground_truth_frames.shape[1] - args.context_window
        effective_steps = min(effective_steps, max_possible_steps)
        if max_possible_steps < args.generation_steps:
            print(f"[WARN] Clamping generation_steps to {effective_steps} (GT has {ground_truth_frames.shape[1]} frames).")

    generated_frames = ground_truth_frames[:, :args.context_window].clone()
    context_frames = ground_truth_frames[:, :args.context_window]

    for i in range(effective_steps):
        print(f"  step {i+1}/{effective_steps}")
        if args.teacher_forced:
            context_frames = ground_truth_frames[:, i:i + args.context_window]
        else:
            context_frames = generated_frames[:, -args.context_window:]

        video_indices = video_tokenizer.tokenize(context_frames)
        video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_indices)

        _sampled_idx, action_latent = get_action_latent(
            cfg, inferred_actions, n_actions, context_frames, latent_action_model, i
        )

        def idx_to_latents(idx):
            return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

        with torch.amp.autocast('cuda', enabled=False):
            next_video_latents = dynamics_model.forward_inference(
                context_latents=video_latents,
                prediction_horizon=args.prediction_horizon,
                num_steps=10,
                index_to_latents_fn=idx_to_latents,
                conditioning=action_latent,
                temperature=args.temperature,
            )
        next_frames = video_tokenizer.detokenize(next_video_latents)
        generated_frames = torch.cat(
            [generated_frames, next_frames[:, -args.prediction_horizon:]],
            dim=1,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # `visualize_inference` writes into a fixed "inference_results" folder under cwd.
    # Patch cwd to redirect output, then restore.
    prev_cwd = Path.cwd()
    os.chdir(out_dir.parent)
    try:
        # Patch save_dir baked into visualize_inference: we already cd'd to the
        # parent so it'll write to <out_dir.parent>/inference_results which we
        # rename below. The hardcoded name is "inference_results".
        visualize_inference(
            generated_frames,
            ground_truth_frames[:, : args.context_window + effective_steps * args.prediction_horizon],
            inferred_actions,
            args.fps,
            use_actions=use_actions,
        )
    finally:
        os.chdir(prev_cwd)

    # Move/rename the freshly-written "inference_results/*" into our labeled out_dir
    tmp_dir = out_dir.parent / "inference_results"
    if tmp_dir.exists():
        for f in tmp_dir.iterdir():
            f.rename(out_dir / f.name)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    # cleanup model to free GPU memory before next ckpt
    del dynamics_model
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    use_actions = not args.no_actions
    set_seed(args.seed)

    # Load VT + LAM once
    print(f"Loading video tokenizer: {args.vt}")
    video_tokenizer, _ = load_videotokenizer_from_checkpoint(args.vt, args.device)
    video_tokenizer.eval()

    latent_action_model = None
    if use_actions:
        print(f"Loading latent actions: {args.lam}")
        latent_action_model, _ = load_latent_actions_from_checkpoint(args.lam, args.device)
        latent_action_model.eval()

    # Load dataset once
    frames_to_load = args.context_window + args.generation_steps * args.prediction_horizon
    print(f"Loading dataset {args.dataset} (frames_to_load={frames_to_load}, preload_ratio={args.preload_ratio})")
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset,
        batch_size=1,
        num_frames=frames_to_load,
        preload_ratio=args.preload_ratio,
    )

    # Pick a single sample deterministically and reuse across checkpoints
    set_seed(args.seed)
    random_idx = random.randint(0, len(data_loader.dataset) - 1)
    print(f"Using dataset sample index {random_idx} of {len(data_loader.dataset)}")
    ground_truth_frames = data_loader.dataset[random_idx][0].unsqueeze(0).to(args.device)
    print(f"GT frames shape: {tuple(ground_truth_frames.shape)}")

    run_label = args.run_label or f"ckpts_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root = Path(args.out_root) / run_label
    print(f"Outputs -> {out_root.resolve()}")

    for step in args.steps:
        dyn_ckpt = Path(args.dyn_root) / f"dynamics_step_{step}"
        if not dyn_ckpt.exists():
            print(f"[SKIP] missing: {dyn_ckpt}")
            continue
        print(f"\n=== Dynamics step {step} ({dyn_ckpt}) ===")
        out_dir = out_root / f"dynamics_step_{step}"
        run_one_checkpoint(
            str(dyn_ckpt),
            args,
            video_tokenizer,
            latent_action_model,
            ground_truth_frames,
            use_actions,
            out_dir,
        )

    print(f"\nAll done. Outputs in {out_root.resolve()}")


if __name__ == "__main__":
    main()
