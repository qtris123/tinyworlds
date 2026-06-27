#!/usr/bin/env bash
# Parallel version of run_eval_sweep.sh: pins one dynamics checkpoint per GPU
# (single process per GPU) and runs both teacher-forced and free-running modes
# back-to-back on that GPU. With 4 GPUs and 4 ckpts, the whole sweep finishes
# in roughly the wall-time of two sequential evaluations rather than eight.
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

VT="${VT:-results/2026_06_08_12_51_05_zelda/video_tokenizer/checkpoints/video_tokenizer_step_37500}"
LAM="${LAM:-results/large_action_model}"
DYN_ROOT="${DYN_ROOT:-results/large_dynamics_model/dynamics/checkpoints}"

# STEPS can be overridden via the env var EVAL_STEPS as a space-separated list,
# e.g. EVAL_STEPS="5000 10000 20000 29000".
read -r -a STEPS <<< "${EVAL_STEPS:-2000 10000 18000 24000}"
MODES=(teacher_forced free_running)
N_GPUS=${N_GPUS:-4}

# All result/log paths are written under this prefix so multiple sweeps (e.g.
# large vs mid dynamics) live side-by-side without colliding.
OUT_PREFIX="${OUT_PREFIX:-evaluation_results}"
mkdir -p "${OUT_PREFIX}"
LOG_DIR="${OUT_PREFIX}/_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# One worker per GPU: walks a list of (step, mode) tasks pinned to that GPU.
run_worker() {
    local gpu_id="$1"
    local step="$2"
    for mode in "${MODES[@]}"; do
        if [[ "$mode" == "teacher_forced" ]]; then tf_flag=true; else tf_flag=false; fi
        local run_dir="${OUT_PREFIX}/${mode}/dynamics_step_${step}"
        local log_file="${LOG_DIR}/gpu${gpu_id}_${mode}_step${step}.log"
        if [[ -f "${run_dir}/ZELDA.json" ]]; then
            echo "[gpu${gpu_id}] skip (already exists): ${run_dir}" | tee -a "${log_file}"
            continue
        fi
        echo "[gpu${gpu_id}] starting ${mode} @ step ${step} -> ${log_file}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/evaluate.py \
            --config configs/evaluation.yaml -- \
            use_latest_checkpoints=false \
            video_tokenizer_path="${VT}" \
            latent_actions_path="${LAM}" \
            dynamics_path="${DYN_ROOT}/dynamics_step_${step}" \
            eval_datasets=[ZELDA] \
            context_window=4 \
            T_pred=8 \
            zelda_use_holdout=true \
            n_clips_per_dataset=32 \
            batch_size=8 \
            n_random_seeds=5 \
            n_visualization_clips=4 \
            teacher_forced="${tf_flag}" \
            output_dir="${OUT_PREFIX}" \
            run_name="${mode}/dynamics_step_${step}" \
            >"${log_file}" 2>&1
        echo "[gpu${gpu_id}] done    ${mode} @ step ${step}"
    done
}

# Launch one worker per GPU and wait for all of them.
pids=()
for i in "${!STEPS[@]}"; do
    gpu=$(( i % N_GPUS ))
    step="${STEPS[$i]}"
    run_worker "${gpu}" "${step}" &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then fail=$((fail + 1)); fi
done

echo
echo "All workers finished (failed=${fail}). Logs in ${LOG_DIR}"
exit "${fail}"
