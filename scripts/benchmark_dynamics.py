"""
Benchmark dynamics training throughput and GPU utilization.

Runs N steps at increasing batch sizes to find the optimal.
Prints: step time, samples/sec, GPU mem used, GPU util%, and recommended LR.

Usage:
    conda run -n tiny python scripts/benchmark_dynamics.py \
        --vt_path results/2026_06_08_12_51_05_zelda/video_tokenizer/checkpoints/video_tokenizer_step_37500 \
        --lam_path results/2026_06_08_12_51_05_zelda/latent_actions/checkpoints/latent_actions_step_9500 \
        --dataset ZELDA
"""

import sys, os, time, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.utils.checkpoint as checkpoint_utils
import subprocess
import yaml

from models.dynamics import DynamicsModel
from utils.utils import load_videotokenizer_from_checkpoint, load_latent_actions_from_checkpoint
from datasets.data_utils import load_data_and_data_loaders

# ── config ──────────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def gpu_stats(device_idx=0):
    """Returns (used_mib, total_mib, util_pct) via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device_idx}",
             "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return 0, 0, 0

def fmt_mib(mib):
    return f"{mib/1024:.1f}GB" if mib >= 1024 else f"{mib}MiB"

# ── benchmark loop ───────────────────────────────────────────────────────────

def run_benchmark(batch_size, dyn_cfg, shared_cfg, vt, lam, device, all_data, n_steps):
    """Run one benchmark trial for a given batch_size. all_data is a pre-loaded tensor [N, T, C, H, W]."""
    patch_size = shared_cfg['patch_size']
    frame_size = shared_cfg['frame_size']
    latent_dim = shared_cfg['latent_dim']
    num_bins   = shared_cfg['num_bins']
    action_dim = lam.action_dim
    n_samples_total = all_data.shape[0]

    dyn = DynamicsModel(
        frame_size=(frame_size, frame_size),
        patch_size=patch_size,
        embed_dim=dyn_cfg['embed_dim'],
        num_heads=dyn_cfg['num_heads'],
        hidden_dim=dyn_cfg['hidden_dim'],
        num_blocks=dyn_cfg['num_blocks'],
        conditioning_dim=action_dim,
        latent_dim=latent_dim,
        num_bins=num_bins,
        use_moe=False,
    ).to(device).train()

    n_params = sum(p.numel() for p in dyn.parameters())
    opt = torch.optim.AdamW(dyn.parameters(), lr=1e-4)
    train_ctx = torch.amp.autocast(device, enabled=True, dtype=torch.bfloat16)

    WARMUP = 3
    times  = []
    peak_mem = 0
    start_idx = 0

    for step in range(WARMUP + n_steps):
        # pull a batch from pre-loaded CPU tensor
        end_idx = min(start_idx + batch_size, n_samples_total)
        x = all_data[start_idx:end_idx].to(device, non_blocking=True)
        start_idx = end_idx if end_idx < n_samples_total else 0
        if x.shape[0] < batch_size:
            x = all_data[:batch_size].to(device, non_blocking=True)

        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            video_tokens  = vt.tokenize(x)
            video_latents = vt.quantizer.get_latents_from_indices(video_tokens, dim=-1)
            actions       = lam.encode(x)

        with train_ctx:
            _, _, loss = dyn(video_latents, training=True, conditioning=actions, targets=video_tokens)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        mem = torch.cuda.memory_allocated(device) // (1024 * 1024)
        peak_mem = max(peak_mem, mem)
        if step >= WARMUP:
            times.append(elapsed)

    # free dyn + optimizer so next trial starts fresh
    del dyn, opt
    torch.cuda.empty_cache()

    avg_step = sum(times) / len(times)
    sps = batch_size / avg_step

    _, total_mib, util_pct = gpu_stats()
    mem_pct = 100 * peak_mem / total_mib if total_mib > 0 else 0
    steps_per_epoch = math.ceil(n_samples_total / batch_size)

    return {
        'batch_size'      : batch_size,
        'n_params'        : n_params,
        'avg_step_ms'     : avg_step * 1000,
        'samples_per_sec' : sps,
        'peak_mem_mib'    : peak_mem,
        'total_mem_mib'   : total_mib,
        'mem_pct'         : mem_pct,
        'gpu_util_pct'    : util_pct,
        'steps_per_epoch' : steps_per_epoch,
        'n_samples'       : n_samples_total,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vt_path",  required=True)
    parser.add_argument("--lam_path", required=True)
    parser.add_argument("--dataset",  default="ZELDA")
    parser.add_argument("--steps",    type=int, default=20, help="benchmark steps after warmup")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=None,
                        help="batch sizes to sweep (default: auto-sweep from 100 up)")
    args = parser.parse_args()

    device = "cuda"
    dtype  = torch.bfloat16

    # load shared + dynamics configs
    shared_cfg = load_yaml("configs/training.yaml")
    dyn_cfg    = load_yaml("configs/dynamics.yaml")
    # dynamics.yaml overrides shared for model shape params
    for k in ("embed_dim", "num_heads", "hidden_dim", "num_blocks"):
        if k in dyn_cfg:
            shared_cfg[k] = dyn_cfg[k]

    print(f"\n{'='*60}")
    print(f"  DynamicsModel config")
    print(f"  embed_dim={dyn_cfg['embed_dim']}, num_heads={dyn_cfg['num_heads']}, "
          f"hidden_dim={dyn_cfg['hidden_dim']}, num_blocks={dyn_cfg['num_blocks']}")
    print(f"  dataset={args.dataset}")
    print(f"{'='*60}\n")

    # load frozen models
    print("Loading Video Tokenizer...")
    vt, _ = load_videotokenizer_from_checkpoint(args.vt_path, device=device, is_distributed=False)
    vt.eval()
    for p in vt.parameters(): p.requires_grad_(False)

    print("Loading Latent Action Model...")
    lam, _ = load_latent_actions_from_checkpoint(args.lam_path, device=device, is_distributed=False)
    lam.eval()
    for p in lam.parameters(): p.requires_grad_(False)

    # ── pre-load dataset once into CPU RAM ──────────────────────────────────
    print(f"\nPre-loading {args.dataset} dataset into CPU RAM (done once, reused per trial)...")
    _, _, loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, batch_size=2048, num_frames=shared_cfg['context_length'],
        distributed=False, rank=0, world_size=1
    )
    all_data = torch.cat([x for x, _ in loader], dim=0)  # [N, T, C, H, W]
    n_total = all_data.shape[0]
    print(f"Loaded {n_total} samples — shape {tuple(all_data.shape)}\n")

    # ── spatial attn memory estimate (P = patches per frame) ───────────────
    P = (shared_cfg['frame_size'] // shared_cfg['patch_size']) ** 2
    B_T_estimate = 10 * shared_cfg['context_length']  # rough B=10 baseline
    attn_map_mib_per_block = (B_T_estimate * dyn_cfg['num_heads'] * P * P * 4) / (1024**2)
    print(f"Spatial attn map size (fp32, P={P}, B×T={B_T_estimate}): "
          f"{attn_map_mib_per_block:.0f} MiB/block × {dyn_cfg['num_blocks']} blocks = "
          f"{attn_map_mib_per_block * dyn_cfg['num_blocks']:.0f} MiB  "
          f"(scales linearly with batch size)\n")

    # ── determine batch sizes to sweep ──────────────────────────────────────
    if args.batch_sizes:
        batch_sizes = args.batch_sizes
    else:
        batch_sizes = [10, 25, 50, 100, 200, 400, 800]

    # ── sweep ──────────────────────────────────────────────────────────────
    print(f"{'Batch':>8} {'Params':>10} {'Step ms':>9} {'Samp/s':>9} "
          f"{'Peak mem':>10} {'Mem%':>6} {'GPU%':>6} {'Steps/ep':>10}")
    print("-" * 80)

    BASE_LR   = dyn_cfg.get('learning_rate', 0.001)
    BASE_BS   = dyn_cfg.get('batch_size_per_gpu', 500)
    best      = None

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            result = run_benchmark(bs, dyn_cfg, shared_cfg, vt, lam, device, all_data, args.steps)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print(f"{bs:>8}  OOM — stopping sweep")
            else:
                print(f"{bs:>8}  Error: {e}")
            break

        # linear LR scaling from base (sqrt scaling is also common for very large BS)
        scaled_lr_linear = BASE_LR * (bs / BASE_BS)
        scaled_lr_sqrt   = BASE_LR * math.sqrt(bs / BASE_BS)

        print(f"{bs:>8} {result['n_params']:>10,} {result['avg_step_ms']:>8.1f}ms "
              f"{result['samples_per_sec']:>9.0f} "
              f"{fmt_mib(result['peak_mem_mib']):>10} {result['mem_pct']:>5.1f}% "
              f"{result['gpu_util_pct']:>5}% {result['steps_per_epoch']:>10,}")
        print(f"          LR: linear={scaled_lr_linear:.2e}  sqrt={scaled_lr_sqrt:.2e}  "
              f"(base={BASE_LR:.2e} @ bs={BASE_BS})")

        best = result
        best['scaled_lr_linear'] = scaled_lr_linear
        best['scaled_lr_sqrt']   = scaled_lr_sqrt

    # ── summary ──
    if best:
        print(f"\n{'='*60}")
        print(f"  Recommended settings  (largest successful batch)")
        print(f"{'='*60}")
        print(f"  batch_size_per_gpu: {best['batch_size']}")
        print(f"  learning_rate (linear scale): {best['scaled_lr_linear']:.2e}")
        print(f"  learning_rate (sqrt scale):   {best['scaled_lr_sqrt']:.2e}")
        print(f"  Peak GPU memory:    {fmt_mib(best['peak_mem_mib'])} / "
              f"{fmt_mib(best['total_mem_mib'])}  ({best['mem_pct']:.1f}%)")
        print(f"  GPU utilization:    {best['gpu_util_pct']}%")
        print(f"  Throughput:         {best['samples_per_sec']:.0f} samples/sec")
        print(f"  Steps per epoch:    {best['steps_per_epoch']:,}  "
              f"({best['n_samples']:,} training samples)")
        print(f"\n  Memory note: P={P} patches → spatial attn is O(B×P²) per block.")
        print(f"  {100-best['mem_pct']:.1f}% VRAM headroom on {best['total_mem_mib']//1024}GB GPU.")

if __name__ == "__main__":
    main()
