"""Genie-style PSNR / Delta_t-PSNR evaluation pipeline.

Loads a trained `(video_tokenizer, latent_action_model, dynamics_model)` suite and
evaluates it on each dataset listed in `configs/evaluation.yaml`. For every dataset
it reports:

  * Tokenizer reconstruction PSNR (the FSQ-VAE-only ceiling)
  * PSNR(x_t, x_hat_t)         — actions inferred from GT frames via the LAM
  * PSNR(x_t, x_hat_prime_t)   — actions sampled uniformly from [0, n_actions),
                                 averaged over `n_random_seeds` seeds
  * Delta_t PSNR per t and the headline number at t = 4 (genie.pdf section 3)

Per-dataset JSON results land in `evaluation_results/eval_<timestamp>/<dataset>.json`,
plus a `summary.md` table comparing all datasets.

Usage
-----

    python scripts/evaluate.py --config configs/evaluation.yaml -- use_latest_checkpoints=true
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.data_utils import load_data_and_data_loaders
from utils.config import EvaluationConfig, load_config
from utils.evaluation_utils import rollout, tokenizer_reconstruct
from utils.inference_utils import load_models
from utils.metrics import frame_psnr
from utils.utils import find_latest_checkpoint, readable_timestamp


def _missing(path) -> bool:
    return (path is None) or (not os.path.exists(path))


def _resolve_checkpoints(args: EvaluationConfig) -> EvaluationConfig:
    base_dir = os.getcwd()
    if args.use_latest_checkpoints or _missing(args.video_tokenizer_path):
        args.video_tokenizer_path = find_latest_checkpoint(base_dir, "video_tokenizer")
    if args.use_latest_checkpoints or _missing(args.latent_actions_path):
        args.latent_actions_path = find_latest_checkpoint(base_dir, "latent_actions")
    if args.use_latest_checkpoints or _missing(args.dynamics_path):
        args.dynamics_path = find_latest_checkpoint(base_dir, "dynamics")

    if _missing(args.video_tokenizer_path):
        raise FileNotFoundError("video_tokenizer_path is unset and no checkpoint was found under results/")
    if _missing(args.latent_actions_path):
        raise FileNotFoundError("latent_actions_path is unset and no checkpoint was found under results/")
    if _missing(args.dynamics_path):
        raise FileNotFoundError("dynamics_path is unset and no checkpoint was found under results/")
    return args


def _build_eval_dataset(dataset_name: str, args: EvaluationConfig, frames_per_clip: int):
    is_zelda = dataset_name == 'ZELDA' # => disable_test_split is False / enable test split
    use_holdout = is_zelda and args.zelda_use_holdout
    # disable_test_split=False splits chronologically 90/10; train half = first 90%, val half = last 10%
    disable_test_split = not use_holdout
    resize_to: Tuple[int, int] = tuple(args.force_resize_to)

    training_data, validation_data, _, _, _ = load_data_and_data_loaders(
        dataset=dataset_name,
        batch_size=args.batch_size,
        num_frames=frames_per_clip,
        disable_test_split=disable_test_split,
        resize_to=resize_to,
    )
    return validation_data if use_holdout else training_data


def _subsample(dataset, n, seed: int):
    N = len(dataset)
    if n is None or n >= N:
        return dataset
    gen = torch.Generator(device='cpu').manual_seed(int(seed))
    perm = torch.randperm(N, generator=gen)[:int(n)].tolist()
    return Subset(dataset, perm)


def _denorm(frames_btchw: torch.Tensor) -> torch.Tensor:
    """Detach + denormalize from [-1, 1] to [0, 1] on CPU as float32."""
    f = frames_btchw.detach().to(torch.float32).cpu()
    return ((f + 1.0) / 2.0).clamp(0.0, 1.0)


def _save_clip_visualization(
    out_dir: str,
    clip_idx: int,
    gt_full: torch.Tensor,           # [T_total, C, H, W]
    x_hat_pred: torch.Tensor,        # [T_pred,  C, H, W]  -- LAM-from-GT actions
    x_hat_prime_pred: torch.Tensor,  # [T_pred,  C, H, W]  -- random actions
    context_window: int,
) -> None:
    """Save a per-clip 3-row PNG: GT / pred(GT-LAM actions) / pred(random actions)."""
    gt = _denorm(gt_full.unsqueeze(0))[0]                 # [T_total, C, H, W]
    pred_gt = _denorm(x_hat_pred.unsqueeze(0))[0]
    pred_rnd = _denorm(x_hat_prime_pred.unsqueeze(0))[0]

    T_total = gt.shape[0]
    T_pred = pred_gt.shape[0]
    assert T_total == context_window + T_pred, (T_total, context_window, T_pred)

    # Prepend the GT context to each prediction row so the eye can see the model's
    # continuation of the same prefix.
    gt_ctx = gt[:context_window]
    pred_gt_full = torch.cat([gt_ctx, pred_gt], dim=0)
    pred_rnd_full = torch.cat([gt_ctx, pred_rnd], dim=0)

    fig, axes = plt.subplots(3, T_total, figsize=(2.0 * T_total, 6.0))
    if T_total == 1:
        axes = axes.reshape(3, 1)
    row_titles = ["Ground truth", "Pred (LAM-inferred actions from GT)", "Pred (random actions)"]
    rows = [gt, pred_gt_full, pred_rnd_full]
    for r, (title, frames) in enumerate(zip(row_titles, rows)):
        for c in range(T_total):
            frame = frames[c].permute(1, 2, 0).numpy()
            ax = axes[r, c]
            ax.imshow(frame)
            ax.set_xticks([]); ax.set_yticks([])
            is_context = c < context_window
            color = 'green' if r == 0 else ('gray' if is_context else ('blue' if r == 1 else 'red'))
            label = ("ctx " if is_context else "pred ") + f"t={c}"
            ax.set_title(label, fontsize=9, color=color)
        axes[r, 0].set_ylabel(title, fontsize=10)

    plt.suptitle(f"clip {clip_idx:03d} -- context_window={context_window}, T_pred={T_pred}", fontsize=12)
    png_path = os.path.join(out_dir, f"clip_{clip_idx:03d}.png")
    plt.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _evaluate_dataset(
    dataset_name: str,
    args: EvaluationConfig,
    video_tokenizer,
    latent_action_model,
    dynamics_model,
    n_actions: int,
    viz_dir: Optional[str] = None,
) -> Dict:
    frames_per_clip = args.context_window + args.T_pred
    eval_dataset = _build_eval_dataset(dataset_name, args, frames_per_clip)
    eval_subset = _subsample(eval_dataset, args.n_clips_per_dataset, args.seed)

    loader = DataLoader(
        eval_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    autocast_dtype = torch.bfloat16 if args.amp else None

    # Sum-then-divide accumulators in float64 for numerical stability across thousands of clips.
    sum_recon_per_t = torch.zeros(frames_per_clip, dtype=torch.float64)
    sum_psnr_gt_per_t = torch.zeros(args.T_pred, dtype=torch.float64)
    sum_psnr_rnd_per_t = torch.zeros(args.T_pred, dtype=torch.float64)
    n_clips = 0

    viz_saved = 0
    n_viz_target = args.n_visualization_clips if viz_dir is not None else 0

    for batch_idx, (x, _) in enumerate(tqdm(loader, desc=f"eval[{dataset_name}]")):
        x = x.to(args.device, non_blocking=True)  # [B, context_window + T_pred, C, H, W]
        B = x.shape[0]
        target = x[:, args.context_window:]  # [B, T_pred, C, H, W]

        with torch.amp.autocast('cuda', enabled=args.amp, dtype=autocast_dtype):
            recon = tokenizer_reconstruct(video_tokenizer, x)  # [B, T_total, C, H, W]
            recon_psnr_bt = frame_psnr(recon, x)
            sum_recon_per_t += recon_psnr_bt.sum(dim=0).double().cpu()

            x_hat = rollout(
                video_tokenizer, latent_action_model, dynamics_model, x,
                action_mode='gt_lam',
                n_actions=n_actions,
                context_window=args.context_window,
                T_pred=args.T_pred,
                prediction_horizon=args.prediction_horizon,
                num_steps=args.num_maskgit_steps,
                temperature=args.temperature,
                teacher_forced=args.teacher_forced,
            )
            psnr_gt_bt = frame_psnr(x_hat, target)  # [B, T_pred]
            sum_psnr_gt_per_t += psnr_gt_bt.sum(dim=0).double().cpu()

            # Average random-action rollouts over n_random_seeds. Combine the global
            # eval seed, the seed index, and batch_idx so different batches see different
            # random action timelines (important when batch_size < dataset size).
            psnr_rnd_per_seed = []
            x_hat_prime_first: Optional[torch.Tensor] = None
            for s in range(args.n_random_seeds):
                rnd_seed = int(args.seed) + 7919 * (s + 1) + batch_idx
                x_hat_prime = rollout(
                    video_tokenizer, latent_action_model, dynamics_model, x,
                    action_mode='random',
                    n_actions=n_actions,
                    context_window=args.context_window,
                    T_pred=args.T_pred,
                    prediction_horizon=args.prediction_horizon,
                    num_steps=args.num_maskgit_steps,
                    temperature=args.temperature,
                    action_seed=rnd_seed,
                    teacher_forced=args.teacher_forced,
                )
                if s == 0:
                    x_hat_prime_first = x_hat_prime
                psnr_rnd_per_seed.append(frame_psnr(x_hat_prime, target))  # [B, T_pred]
            psnr_rnd_bt = torch.stack(psnr_rnd_per_seed, dim=0).mean(dim=0)  # [B, T_pred]
            sum_psnr_rnd_per_t += psnr_rnd_bt.sum(dim=0).double().cpu()

        # Save visualizations for the first n_viz_target clips of the dataset
        if viz_dir is not None and viz_saved < n_viz_target and x_hat_prime_first is not None:
            n_to_save = min(B, n_viz_target - viz_saved)
            for k in range(n_to_save):
                _save_clip_visualization(
                    out_dir=viz_dir,
                    clip_idx=viz_saved,
                    gt_full=x[k],
                    x_hat_pred=x_hat[k],
                    x_hat_prime_pred=x_hat_prime_first[k],
                    context_window=args.context_window,
                )
                viz_saved += 1

        n_clips += B

    if n_clips == 0:
        raise RuntimeError(f"No clips found for dataset {dataset_name}; check preload_ratio / holdout split")

    recon_per_t = (sum_recon_per_t / n_clips).tolist()
    psnr_gt_per_t = (sum_psnr_gt_per_t / n_clips).tolist()
    psnr_rnd_per_t = (sum_psnr_rnd_per_t / n_clips).tolist()
    delta_psnr_per_t = [g - r for g, r in zip(psnr_gt_per_t, psnr_rnd_per_t)]
    delta_at_t4 = delta_psnr_per_t[3] if args.T_pred >= 4 else None

    return {
        'dataset': dataset_name,
        'n_clips': n_clips,
        'context_window': args.context_window,
        'T_pred': args.T_pred,
        'prediction_horizon': args.prediction_horizon,
        'num_maskgit_steps': args.num_maskgit_steps,
        'temperature': args.temperature,
        'n_random_seeds': args.n_random_seeds,
        'n_actions': n_actions,
        'force_resize_to': list(args.force_resize_to),
        'used_zelda_holdout': dataset_name == 'ZELDA' and args.zelda_use_holdout,
        'teacher_forced': args.teacher_forced,
        'tokenizer_recon_psnr_per_t': recon_per_t,
        'tokenizer_recon_psnr_mean': float(sum(recon_per_t) / max(len(recon_per_t), 1)),
        'psnr_gt_actions_per_t': psnr_gt_per_t,
        'psnr_random_actions_per_t': psnr_rnd_per_t,
        'delta_psnr_per_t': delta_psnr_per_t,
        'delta_psnr_at_t4': delta_at_t4,
    }


def _format_summary_table(results: List[Dict], args: EvaluationConfig) -> str:
    lines: List[str] = []
    lines.append("# Evaluation summary\n")
    lines.append(f"- Checkpoints: VT={args.video_tokenizer_path}, LAM={args.latent_actions_path}, Dyn={args.dynamics_path}")
    lines.append(f"- context_window={args.context_window}, T_pred={args.T_pred}, "
                 f"num_maskgit_steps={args.num_maskgit_steps}, temperature={args.temperature}, "
                 f"n_random_seeds={args.n_random_seeds}\n")
    lines.append("| Dataset | N clips | Recon PSNR (mean) | PSNR(x,x_hat) t=4 | PSNR(x,x_hat') t=4 | Delta_4 PSNR |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        recon = r['tokenizer_recon_psnr_mean']
        psnr_gt = r['psnr_gt_actions_per_t'][3] if len(r['psnr_gt_actions_per_t']) >= 4 else float('nan')
        psnr_rnd = r['psnr_random_actions_per_t'][3] if len(r['psnr_random_actions_per_t']) >= 4 else float('nan')
        delta = r['delta_psnr_at_t4'] if r['delta_psnr_at_t4'] is not None else float('nan')
        lines.append(
            f"| {r['dataset']} | {r['n_clips']} | {recon:.3f} | "
            f"{psnr_gt:.3f} | {psnr_rnd:.3f} | {delta:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    args: EvaluationConfig = load_config(
        EvaluationConfig,
        default_config_path=os.path.join(os.getcwd(), 'configs', 'evaluation.yaml'),
    )

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args = _resolve_checkpoints(args)
    print(f"Using video_tokenizer checkpoint: {args.video_tokenizer_path}")
    print(f"Using latent_actions  checkpoint: {args.latent_actions_path}")
    print(f"Using dynamics        checkpoint: {args.dynamics_path}")

    video_tokenizer, latent_action_model, dynamics_model = load_models(
        args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path,
        args.device, use_actions=True,
    )
    n_actions = int(latent_action_model.quantizer.codebook_size)
    print(f"Detected n_actions = {n_actions}")

    if args.T_pred < 4:
        print(f"[WARN] T_pred={args.T_pred} < 4; the paper's headline Delta_t PSNR uses t=4 and will be reported as null.")

    run_subdir = args.run_name if args.run_name else f"eval_{readable_timestamp()}"
    out_root = os.path.join(os.getcwd(), args.output_dir, run_subdir)
    os.makedirs(out_root, exist_ok=True)
    print(f"Writing results to: {out_root}")

    results: List[Dict] = []
    for dataset_name in args.eval_datasets:
        print(f"\n===== Evaluating on {dataset_name} =====")
        viz_dir: Optional[str] = None
        if args.n_visualization_clips > 0:
            viz_dir = os.path.join(out_root, dataset_name, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

        result = _evaluate_dataset(
            dataset_name, args,
            video_tokenizer, latent_action_model, dynamics_model,
            n_actions=n_actions,
            viz_dir=viz_dir,
        )
        results.append(result)

        json_path = os.path.join(out_root, f"{dataset_name}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {json_path}")
        print(f"  tokenizer recon PSNR (mean over t): {result['tokenizer_recon_psnr_mean']:.3f}")
        if args.T_pred >= 4:
            print(f"  PSNR(x_t, x_hat_t)         @ t=4: {result['psnr_gt_actions_per_t'][3]:.3f}")
            print(f"  PSNR(x_t, x_hat_prime_t)   @ t=4: {result['psnr_random_actions_per_t'][3]:.3f}")
            print(f"  Delta_4 PSNR               @ t=4: {result['delta_psnr_at_t4']:+.3f}")

    summary_md = _format_summary_table(results, args)
    summary_path = os.path.join(out_root, "summary.md")
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    print("\n" + summary_md)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
