from contextlib import nullcontext
import torch
import os
from tqdm import tqdm
from einops import rearrange
from models.dynamics import DynamicsModel
from datasets.data_utils import visualize_reconstruction, load_data_and_data_loaders, load_eval_data_and_loader
from tqdm import tqdm
from einops import rearrange
from utils.wandb_utils import (
    init_wandb, log_training_metrics, log_learning_rate, log_system_metrics, finish_wandb, log_action_distribution, log_data_partition,
    clip_and_log_grad_norm,
)
from utils.scheduler_utils import create_cosine_scheduler
from utils.utils import (
    readable_timestamp,
    save_training_state,
    load_videotokenizer_from_checkpoint,
    load_latent_actions_from_checkpoint,
    load_dynamics_from_checkpoint,
    prepare_pipeline_run_root,
    prepare_stage_dirs,
)
from utils.config import DynamicsConfig, load_stage_config_merged
import wandb
import yaml
from dataclasses import asdict
from utils.distributed import init_distributed_from_env, prepare_model_for_distributed, unwrap_model, print_param_count_if_main, cleanup_distributed
from torch.distributed.fsdp import FSDPModule

def main():
    # dynamics config merged with training_config.yaml (training takes priority), plus CLI overrides
    args: DynamicsConfig = load_stage_config_merged(DynamicsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'dynamics.yaml'))
    # DDP setup
    dist_setup = init_distributed_from_env()

    # run save dir if it doesn't exist (running not from full train)
    is_main = dist_setup['is_main']
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'dynamics')
    if is_main:
        print(f"Dynamics Training")
        print(f"Results will be saved in {stage_dir}")

    # load video tokenizer and latent action model.
    # strict=False here tolerates frozen checkpoints saved before QK-Norm was added
    # to STTransformer — the new q_norm/k_norm scale params keep their constructor
    # init (≈ identity), which matches how these models behaved during their
    # original training. Both modules are then put into eval() and frozen.
    if os.path.isdir(args.video_tokenizer_path):
        video_tokenizer, vq_ckpt = load_videotokenizer_from_checkpoint(
            checkpoint_path=args.video_tokenizer_path, 
            device=args.device, 
            is_distributed=dist_setup['is_distributed'],
            strict=False,
        )
        video_tokenizer.eval()
        for p in video_tokenizer.parameters():
            p.requires_grad = False
    else:
        raise FileNotFoundError(f"Video tokenizer checkpoint not found at {args.video_tokenizer_path}")
    if os.path.isdir(args.latent_actions_path):
        latent_action_model, latent_action_ckpt = load_latent_actions_from_checkpoint(
            checkpoint_path=args.latent_actions_path, 
            device=args.device,
            is_distributed=dist_setup['is_distributed'],
            strict=False,
        )
        unwrap_model(latent_action_model).eval()
        for p in unwrap_model(latent_action_model).parameters():
            p.requires_grad = False
    else:
        raise FileNotFoundError(f"Latent Action Model checkpoint not found at {args.latent_actions_path}")

    # init dynamics model and optional ckpt load
    dynamics_model = DynamicsModel(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        conditioning_dim=unwrap_model(latent_action_model).action_dim,
        latent_dim=args.latent_dim,
        num_bins=args.num_bins,
        use_moe=getattr(args, 'use_moe', False),
        num_experts=getattr(args, 'num_experts', 4),
        top_k_experts=getattr(args, 'top_k_experts', 2),
        moe_aux_loss_coeff=getattr(args, 'moe_aux_loss_coeff', 0.01),
    ).to(args.device)
    if args.checkpoint:
        # strict=False lets us load a pre-QK-Norm checkpoint into the current
        # model (the new q_norm/k_norm params keep their constructor init).
        dynamics_model, _ = load_dynamics_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=args.device,
            model=dynamics_model,
            is_distributed=dist_setup['is_distributed'],
            strict=False,
        )

    # optional DDP, compile, param count, tf32
    print_param_count_if_main(dynamics_model, "DynamicsModel", is_main)
    if args.compile:
        video_tokenizer = torch.compile(video_tokenizer, mode="reduce-overhead", fullgraph=False, dynamic=True)
        latent_action_model = torch.compile(latent_action_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        dynamics_model = torch.compile(dynamics_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        print("Compiled all models for training.")
    dynamics_model = prepare_model_for_distributed(
        dynamics_model, 
        args.distributed, 
        model_type=dynamics_model.model_type,
        device_mesh=dist_setup['device_mesh'],
    )
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # create optimizer(s) — AdamW or Muon+AdamW split
    from utils.optimizer_utils import create_optimizer
    optimizers = create_optimizer(dynamics_model, args)

    # cosine scheduler for lr warmup and AMP grad scaler
    schedulers = [create_cosine_scheduler(opt, args.n_updates) for opt in optimizers]
    train_ctx = torch.amp.autocast(args.device, enabled=True, dtype=torch.bfloat16) if args.amp and not args.distributed.use_fsdp else nullcontext()

    results = {
        'n_updates': 0,
        'dynamics_losses': [],
        'loss_vals': [],
    }

    # init wandb
    if args.use_wandb and is_main:
        run_name = f"dynamics_{readable_timestamp()}"
        init_wandb(args.wandb_project, asdict(args), run_name)

    unwrap_model(dynamics_model).train()

    # dataloader
    enable_eval_loss = getattr(args, 'enable_eval_loss', False)
    train_ratio = getattr(args, 'train_ratio', 0.8)
    eval_ratio = getattr(args, 'eval_ratio', 0.1)
    eval_loss_interval = getattr(args, 'eval_loss_interval', 0) or args.log_interval

    data_overrides = {}
    if hasattr(args, 'fps') and args.fps is not None:
        data_overrides['fps'] = args.fps
    if hasattr(args, 'preload_ratio') and args.preload_ratio is not None:
        data_overrides['preload_ratio'] = args.preload_ratio

    partition_kwargs = {}
    if enable_eval_loss:
        partition_kwargs = dict(enable_partition=True, partition='train', train_ratio=train_ratio, eval_ratio=eval_ratio)

    training_data, _, training_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size_per_gpu,
        num_frames=args.context_length,
        distributed=dist_setup['is_distributed'],
        rank=dist_setup['device_mesh'].get_rank() if dist_setup['device_mesh'] is not None else 0,
        world_size=dist_setup['world_size'],
        **data_overrides,
        **partition_kwargs,
    )

    eval_data = None
    eval_loader = None
    if enable_eval_loss:
        eval_data, eval_loader = load_eval_data_and_loader(
            dataset=args.dataset,
            batch_size=args.batch_size_per_gpu,
            num_frames=args.context_length,
            train_ratio=train_ratio,
            eval_ratio=eval_ratio,
            **data_overrides,
        )

    # Write data partition settings file on main process.
    # total_raw_frames is set before partition slicing, so it reflects the full dataset.
    if is_main:
        full_raw = training_data.total_raw_frames
        test_ratio = max(0.0, 1.0 - train_ratio - eval_ratio) if enable_eval_loss else 0.0
        train_frames = int(len(training_data.data))
        eval_frames = int(len(eval_data.data)) if eval_data is not None else 0
        partition_info = {
            'enable_eval_loss': enable_eval_loss,
            'train_ratio': train_ratio if enable_eval_loss else 1.0,
            'eval_ratio': eval_ratio if enable_eval_loss else 0.0,
            'test_ratio': round(test_ratio, 6),
            'raw_frames_total': full_raw,
            'train_frames': train_frames,
            'eval_frames': eval_frames,
            'test_frames': max(0, full_raw - train_frames - eval_frames),
            'train_clips': len(training_data),
            'eval_clips': len(eval_data) if eval_data is not None else 0,
            'frame_skip': training_data.frame_skip,
            'num_frames': args.context_length,
            'dataset': args.dataset,
        }
        partition_file = os.path.join(stage_dir, 'data_partition.yaml')
        with open(partition_file, 'w') as f:
            yaml.dump(partition_info, f, default_flow_style=False, sort_keys=False)
        print(f"Data partition info written to {partition_file}")
        if args.use_wandb:
            log_data_partition(partition_info)

    train_iter = iter(training_loader)

    use_moe = getattr(args, 'use_moe', False)

    for i in tqdm(range(0, args.n_updates), disable=not is_main):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if isinstance(dynamics_model, FSDPModule):
            dynamics_model.set_requires_gradient_sync(False)
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        for micro_batch in range(args.gradient_accumulation_steps):
            try:
                x, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(training_loader)  # reset iterator when epoch ends
                x, _ = next(train_iter)

            x = x.to(args.device, non_blocking=True)  # [batch_size, seq_len, channels, height, width]

            # VT and LAM are frozen — wrap their forwards in no_grad so PyTorch
            # never holds onto their intermediate activations. With the large LAM
            # (67M, 8-block, embed=512) this matters: keeping its forward graph
            # alive blows the 96 GiB GH200 budget once combined with the 18-block
            # dynamics transformer's own activation memory.
            with torch.no_grad():
                video_tokens = video_tokenizer.tokenize(x) # [B, T, P]
                video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_tokens, dim=-1) # [B, T, P, L]
                if args.use_actions:
                    quantized_actions = latent_action_model.encode(x)  # [B, T - 1, A]
                else:
                    quantized_actions = None

            # predict masked frame latents with dynamics model (masking in dynamics model)
            with train_ctx:
                predicted_next_logits, mask_positions, loss = dynamics_model(
                    video_latents, training=True, conditioning=quantized_actions, targets=video_tokens
                )

                # add MoE load-balancing auxiliary loss
                moe_aux_scalar = 0.0
                if use_moe:
                    aux_loss = unwrap_model(dynamics_model).transformer.moe_aux_loss()
                    moe_aux_scalar = aux_loss.item()
                    loss = loss + aux_loss

                if isinstance(dynamics_model, FSDPModule):
                    if (micro_batch + 1) % args.gradient_accumulation_steps == 0:
                        dynamics_model.set_requires_gradient_sync(True)

                loss /= args.gradient_accumulation_steps
                loss.backward()

        results['n_updates'] = i
        results['dynamics_losses'].append(loss.detach().cpu())
        results['loss_vals'].append(loss.detach().cpu())

        # Clip gradients, capture pre-clip norm, log to wandb, and skip the
        # optimizer step on non-finite grads so one bad batch cannot poison
        # AdamW's second-moment estimates.
        pre_clip_norm_val, nonfinite_grad = clip_and_log_grad_norm(
            unwrap_model(dynamics_model).parameters(),
            max_norm=1.0,
            step=i,
            log_to_wandb=(args.use_wandb and is_main),
        )

        if nonfinite_grad:
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            if is_main:
                print(f"\n[WARN] non-finite grad-norm at step {i} (norm={pre_clip_norm_val}); skipping optimizer step")
        else:
            for opt in optimizers:
                opt.step()
        for sched in schedulers:
            sched.step()

        # Memory probe — opt-in via NG_MEMPROBE=1, prints peak/current memory
        # for every rank at the end of every step.  Cheap; gated so it stays out
        # of normal training output. Reset stats at step 1 so we don't measure
        # the one-shot startup allocation peak.
        if os.environ.get('NG_MEMPROBE') == '1':
            if i == 1:
                torch.cuda.reset_peak_memory_stats()
            if i > 0:
                rank = dist_setup['device_mesh'].get_rank() if dist_setup['device_mesh'] is not None else 0
                peak_alloc = torch.cuda.max_memory_allocated() / 1e9
                peak_reserved = torch.cuda.max_memory_reserved() / 1e9
                curr_alloc = torch.cuda.memory_allocated() / 1e9
                print(
                    f"[memprobe rank={rank} step={i:04d}] "
                    f"peak_alloc={peak_alloc:6.2f} GiB  "
                    f"peak_reserved={peak_reserved:6.2f} GiB  "
                    f"curr_alloc={curr_alloc:6.2f} GiB",
                    flush=True,
                )

        # wandb logging
        if args.use_wandb and is_main:
            log_dict = {'train/loss': loss.item()}
            if use_moe:
                log_dict['train/moe_aux_loss'] = moe_aux_scalar
                # per-expert utilization across blocks
                expert_util = unwrap_model(dynamics_model).transformer.moe_expert_utilization()
                for block_idx, fracs in expert_util.items():
                    for expert_i, frac in enumerate(fracs.tolist()):
                        log_dict[f'moe/block{block_idx}_expert{expert_i}_frac'] = frac
            wandb.log(log_dict, step=i)
            log_system_metrics(i)
            log_learning_rate(optimizers[0], i)
            if args.use_actions:
                action_indices = latent_action_model.quantizer.get_indices_from_latents(quantized_actions)
                log_action_distribution(action_indices, i, args.n_actions)

        # eval loss — computed before checkpoint so both metrics land at the same step.
        # NOTE: we deliberately keep dynamics_model in train mode (no .eval()/.train()
        # toggle) because DynamicsModel.forward gates the masking and loss path on
        # `training and self.training` — flipping to eval() would short-circuit the
        # loss path and return None. The model has no dropout/BN so train-mode is
        # numerically identical here, and `torch.no_grad()` is what actually prevents
        # gradient computation / autograd-graph retention.
        if enable_eval_loss and eval_loader is not None and (i % eval_loss_interval == 0):
            eval_loss_vals = []
            with torch.no_grad():
                for x_eval, _ in eval_loader:
                    x_eval = x_eval.to(args.device, non_blocking=True)
                    eval_video_tokens = video_tokenizer.tokenize(x_eval)
                    eval_video_latents = video_tokenizer.quantizer.get_latents_from_indices(eval_video_tokens, dim=-1)
                    if args.use_actions:
                        eval_quantized_actions = latent_action_model.encode(x_eval)
                    else:
                        eval_quantized_actions = None
                    with train_ctx:
                        _, _, eval_loss_batch = dynamics_model(
                            eval_video_latents, training=True,
                            conditioning=eval_quantized_actions, targets=eval_video_tokens,
                        )
                    eval_loss_vals.append(eval_loss_batch.item())
            if eval_loss_vals and is_main:
                mean_eval_loss = sum(eval_loss_vals) / len(eval_loss_vals)
                if args.use_wandb:
                    wandb.log({'eval/loss': mean_eval_loss}, step=i)
                print(f'\n Step {i} Eval Loss: {mean_eval_loss:.6f}')

        # save model and visualize results
        if i % args.log_interval == 0:
            # decode predictions and build masked-frame overlay for visualization (always, not just wandb)
            predicted_next_indices = torch.argmax(predicted_next_logits, dim=-1)
            predicted_next_latents = video_tokenizer.quantizer.get_latents_from_indices(predicted_next_indices, dim=-1)
            with torch.no_grad():
                predicted_frames = video_tokenizer.decoder(predicted_next_latents[:16])  # [B, T, C, H, W]

            B, T, P = mask_positions.shape
            patch_size = args.patch_size
            H, W = args.frame_size, args.frame_size
            pixel_mask = torch.zeros(B, T, H, W, device=mask_positions.device)
            for b in range(B):
                for t in range(T):
                    for p in range(P):
                        if mask_positions[b, t, p]:
                            patch_row = (p // (W // patch_size)) * patch_size
                            patch_col = (p % (W // patch_size)) * patch_size
                            pixel_mask[b, t, patch_row:patch_row+patch_size, patch_col:patch_col+patch_size] = 1
            pixel_mask_expanded = rearrange(pixel_mask, 'b t h w -> b t 1 h w')
            masked_frames = x * (1 - pixel_mask_expanded)

            hyperparameters = args.__dict__
            ckpt_path = save_training_state(dynamics_model, optimizers[0], schedulers[0], hyperparameters, checkpoints_dir, prefix='dynamics', step=i)
            # save secondary optimizer/scheduler state when using split optimizers (Muon+AdamW).
            # Gate to rank 0: otherwise every DDP rank races on the same file
            # path, same bug pattern that wiped 28/30 model_state_dict.pt files
            # in 2026_06_23_09_31_58_30k (see save_training_state docstring).
            if len(optimizers) > 1 and is_main:
                import pathlib
                torch.save(
                    {f'optimizer_{j}': o.state_dict() for j, o in enumerate(optimizers)},
                    pathlib.Path(ckpt_path) / 'all_optimizers.pt',
                )
                torch.save(
                    {f'scheduler_{j}': s.state_dict() for j, s in enumerate(schedulers)},
                    pathlib.Path(ckpt_path) / 'all_schedulers.pt',
                )
            if is_main:
                save_path = os.path.join(visualizations_dir, f'dynamics_prediction_step_{i}.png')
                vis_actions = None
                if args.use_actions and quantized_actions is not None:
                    vis_actions = latent_action_model.quantizer.get_indices_from_latents(
                        quantized_actions[:16].detach()
                    ).cpu()
                visualize_reconstruction(masked_frames[:16].cpu(), predicted_frames[:16].cpu(), save_path, actions=vis_actions)

            print('\n Step', i, 'Loss:', torch.mean(torch.stack(results["loss_vals"][-args.log_interval:])).item())

    # finish wandb
    if args.use_wandb and is_main:
        finish_wandb()
    cleanup_distributed(dist_setup['is_distributed'])

if __name__ == "__main__":
    main()
