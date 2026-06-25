#!/usr/bin/env bash
# launch_dynamics_4gpu.sh — fresh dynamics-only training on 4× GH200 with the large LAM.
#
# Wires:
#   * VT  = results/2026_06_08_12_51_05_zelda/video_tokenizer/checkpoints/video_tokenizer_step_37500
#   * LAM = results/large_action_model        (67M Zelda LAM, n_actions=4)
#   * Dataset = ZELDA  (data/zelda_frames.h5)
#
# Configs (committed alongside this script):
#   * configs/training_large_lam.yaml  — 4-GPU DDP, eval-loss enabled, AMP+TF32 on
#   * configs/dynamics_large_lam.yaml  — 18-block / 512-dim / 30k updates, eff_batch=512
#
# Usage:
#   bash scripts/launch_dynamics_4gpu.sh                       # full 30k run
#   N_UPDATES=30 BATCH_SIZE_PER_GPU=128 bash scripts/launch_dynamics_4gpu.sh   # smoke test
#
# Override knobs via env vars:
#   NPROC_PER_NODE        default 4
#   BATCH_SIZE_PER_GPU    default uses dynamics_large_lam.yaml value (128)
#   GRAD_ACCUM_STEPS      default uses dynamics_large_lam.yaml value (1)
#   N_UPDATES             default uses dynamics_large_lam.yaml value (30000)
#   USE_WANDB             default true (run name is always auto: dynamics_<timestamp>)
#   EXTRA_OVERRIDES       extra dotlist overrides, e.g. "learning_rate=0.0005 num_blocks=16"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the dedicated tinyworlds conda env (created via `conda create -n tiny --clone cartridges`).
CONDA_BASE="$(conda info --base 2>/dev/null || echo "/localhome/local-triv/miniconda3")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate tiny

# Ensure tinyworlds is on PYTHONPATH so `import models / utils / datasets` works under torchrun.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# W&B credentials live in a shell function in ~/.bashrc (`tinyworlds_env`) to avoid
# shadowing other projects' WANDB_API_KEY. Call it if it's defined so we pick up the
# right key without requiring the user to source .bashrc beforehand.
if declare -F tinyworlds_env >/dev/null 2>&1; then
  tinyworlds_env >/dev/null
fi

# Configurable overrides (default to YAML values when unset).
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
USE_WANDB="${USE_WANDB:-true}"

OVERRIDES=()
[[ -n "${BATCH_SIZE_PER_GPU:-}" ]] && OVERRIDES+=("batch_size_per_gpu=${BATCH_SIZE_PER_GPU}")
[[ -n "${GRAD_ACCUM_STEPS:-}"   ]] && OVERRIDES+=("gradient_accumulation_steps=${GRAD_ACCUM_STEPS}")
[[ -n "${N_UPDATES:-}"          ]] && OVERRIDES+=("n_updates=${N_UPDATES}")
[[ -n "${LEARNING_RATE:-}"      ]] && OVERRIDES+=("learning_rate=${LEARNING_RATE}")
[[ -n "${LOG_INTERVAL:-}"       ]] && OVERRIDES+=("log_interval=${LOG_INTERVAL}")
OVERRIDES+=("use_wandb=${USE_WANDB}")

# Any extra free-form dotlist overrides the caller wants to pass through.
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=( ${EXTRA_OVERRIDES} )
  OVERRIDES+=( "${EXTRA_ARR[@]}" )
fi

# Reduce CUDA allocator fragmentation. Without this, DDP+AMP on the 18-block / 512-dim
# / 2048-hidden dynamics transformer hits "reserved-but-unallocated" pressure under
# the 96 GiB GH200 budget and OOMs at smaller batch sizes than necessary.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[launch_dynamics_4gpu] cwd=$REPO_ROOT"
echo "[launch_dynamics_4gpu] python=$(which python)"
echo "[launch_dynamics_4gpu] nproc_per_node=${NPROC_PER_NODE}"
echo "[launch_dynamics_4gpu] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[launch_dynamics_4gpu] overrides: ${OVERRIDES[*]:-<none>}"

# Reserve the right number of GPUs (visible to torchrun). If CUDA_VISIBLE_DEVICES is set
# upstream we respect it; otherwise default to GPUs 0..NPROC_PER_NODE-1.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  vis=""
  for ((i=0; i<NPROC_PER_NODE; i++)); do
    vis+="${i}"
    if (( i < NPROC_PER_NODE - 1 )); then vis+="," ; fi
  done
  export CUDA_VISIBLE_DEVICES="$vis"
fi
echo "[launch_dynamics_4gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/train_dynamics.py \
  --config configs/dynamics_large_lam.yaml \
  --training_config configs/training_large_lam.yaml \
  -- "${OVERRIDES[@]}"
