from contextlib import nullcontext
import torch
import os
from models.latent_actions import LatentActionModel
from datasets.data_utils import load_data_and_data_loaders, load_eval_data_and_loader, visualize_reconstruction
from utils.scheduler_utils import create_cosine_scheduler
from tqdm import tqdm
import wandb
from utils.utils import readable_timestamp, save_training_state, prepare_stage_dirs, prepare_pipeline_run_root
from utils.config import LatentActionsConfig, load_stage_config_merged
import yaml
from utils.utils import save_training_state, load_latent_actions_from_checkpoint
from utils.wandb_utils import init_wandb, log_system_metrics, finish_wandb, log_action_distribution, log_learning_rate, log_data_partition, clip_and_log_grad_norm
from dataclasses import asdict
from utils.distributed import init_distributed_from_env, prepare_model_for_distributed, unwrap_model, print_param_count_if_main, cleanup_distributed
from torch.distributed.fsdp import FSDPModule

def main():
    # latent actions config merged with training_config.yaml (training takes priority), plus CLI overrides
    args: LatentActionsConfig = load_stage_config_merged(LatentActionsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'latent_actions.yaml'))

    # DDP setup
    dist_setup = init_distributed_from_env()

    # run save dir if it doesn't exist (running not from full train)
    timestamp = readable_timestamp()
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    is_main = dist_setup['is_main']
    stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'latent_actions')
    if is_main:
        print(f"Latent Actions Training")
        print(f'Results will be saved in {stage_dir}')

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

    training_data, validation_data, training_loader, validation_loader, x_train_var = load_data_and_data_loaders(
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

    # init model and optional ckpt load
    model = LatentActionModel(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        n_actions=args.n_actions,
    ).to(args.device)
    if args.checkpoint:
        model, _ = load_latent_actions_from_checkpoint(
            args.checkpoint, 
            args.device,
            model,
            dist_setup['is_distributed'],
        )

    # optional DDP, compile, param count, tf32
    print_param_count_if_main(model, "LatentActionModel", is_main)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
    model = prepare_model_for_distributed(
        model, 
        args.distributed, 
        model_type=model.model_type, 
        device_mesh=dist_setup['device_mesh'],
    )
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # create optimizer(s) — AdamW or Muon+AdamW split
    from utils.optimizer_utils import create_optimizer
    optimizers = create_optimizer(model, args)

    # cosine scheduler for lr warmup and AMP
    schedulers = [create_cosine_scheduler(opt, args.n_updates) for opt in optimizers]
    train_ctx = torch.amp.autocast(args.device, enabled=True, dtype=torch.bfloat16) if args.amp and not args.distributed.use_fsdp else nullcontext()

    results = {
        'n_updates': 0,
        'loss_vals': [],
    }

    # init wandb
    if args.use_wandb and is_main:
        cfg = asdict(args)
        cfg.update({'timestamp': timestamp})
        run_name = f"latent_actions_{timestamp}"
        init_wandb(args.wandb_project, cfg, run_name)
        log_data_partition(partition_info)

    unwrap_model(model).train()

    train_iter = iter(training_loader)
    for i in tqdm(range(args.n_updates), disable=not is_main):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if isinstance(model, FSDPModule):
            model.set_requires_gradient_sync(False)
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        for micro_batch in range(args.gradient_accumulation_steps):
            try:
                (x, _) = next(train_iter)
            except StopIteration:
                train_iter = iter(training_loader)
                (x, _) = next(train_iter)

            x = x.to(args.device, non_blocking=True)

            with train_ctx:
                loss, pred_frames = model(x)
                loss /= args.gradient_accumulation_steps
                if isinstance(model, FSDPModule):
                    if (micro_batch + 1) % args.gradient_accumulation_steps == 0:
                        model.set_requires_gradient_sync(True)
                loss.backward()

        pre_clip_norm_val, nonfinite_grad = clip_and_log_grad_norm(
            unwrap_model(model).parameters(),
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

        results['n_updates'] = i
        results['loss_vals'].append(loss.detach().cpu())

        if args.use_wandb and is_main:
            wandb.log({
                'train/loss': loss.item(),
            }, step=i)
            log_system_metrics(i)
            log_learning_rate(optimizers[0], i)
  
        # eval loss — computed before checkpoint so both metrics land at the same step
        if enable_eval_loss and eval_loader is not None and (i % eval_loss_interval == 0):
            model.eval()
            eval_loss_vals = []
            with torch.no_grad():
                for x_eval, _ in eval_loader:
                    x_eval = x_eval.to(args.device, non_blocking=True)
                    with train_ctx:
                        eval_loss_batch, _ = model(x_eval)
                    eval_loss_vals.append(eval_loss_batch.item())
            model.train()
            if eval_loss_vals and is_main:
                mean_eval_loss = sum(eval_loss_vals) / len(eval_loss_vals)
                if args.use_wandb:
                    wandb.log({'eval/loss': mean_eval_loss}, step=i)
                print(f'\n Step {i} Eval Loss: {mean_eval_loss:.6f}')

        # save model and visualize results
        if i % args.log_interval == 0:
            if args.use_wandb:
                with torch.no_grad():
                    actions = unwrap_model(model).encoder(x)
                    actions_quantized = unwrap_model(model).quantizer(actions)
                    idx = unwrap_model(model).quantizer.get_indices_from_latents(actions_quantized)
                    codebook_usage = idx.unique().numel() / unwrap_model(model).quantizer.codebook_size
                    z_e_var = actions.var(dim=0, unbiased=False).mean().item()
                    pred_frames_var = pred_frames.var(dim=0, unbiased=False).mean().item()

            if args.use_wandb and is_main:
                wandb.log({
                    "latent_actions/codebook_usage": codebook_usage,
                    "latent_actions/encoder_variance": z_e_var,
                    "latent_actions/decoder_variance": pred_frames_var,
                }, step=i)
                log_action_distribution(idx, i, args.n_actions)

            hyperparameters = vars(args)
            save_training_state(model, optimizers[0], None, hyperparameters, checkpoints_dir, prefix='latent_actions', step=i)
            if is_main:
                save_path = os.path.join(visualizations_dir, f'reconstructions_latent_actions_step_{i}.png')
                visualize_reconstruction(x, pred_frames, save_path)
            
                print('\n Step', i, 'Loss:', loss.item(), 'Codebook Usage:', codebook_usage, 'Encoder Variance:', z_e_var, 'Decoder Variance:', pred_frames_var)

    # finish wandb
    if args.use_wandb and is_main:
        finish_wandb()
    cleanup_distributed(dist_setup['is_distributed'])

if __name__ == "__main__":
    main()
