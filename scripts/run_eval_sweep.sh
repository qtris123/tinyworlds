#!/usr/bin/env bash
# Run the Genie-style evaluation for several dynamics checkpoints in both
# teacher-forced and free-running rollout modes. Results land in
#   evaluation_results/<mode>/dynamics_step_<step>/
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

VT="results/2026_06_08_12_51_05_zelda/video_tokenizer/checkpoints/video_tokenizer_step_37500"
LAM="results/large_action_model"
DYN_ROOT="results/large_dynamics_model/dynamics/checkpoints"

STEPS=(2000 10000 18000 24000)
MODES=(teacher_forced free_running)

for mode in "${MODES[@]}"; do
    if [[ "$mode" == "teacher_forced" ]]; then
        tf_flag=true
    else
        tf_flag=false
    fi

    for step in "${STEPS[@]}"; do
        run_dir="evaluation_results/${mode}/dynamics_step_${step}"
        if [[ -f "${run_dir}/ZELDA.json" ]]; then
            echo "[skip] ${run_dir} already exists"
            continue
        fi
        echo "==========================================================="
        echo "Evaluating ${mode} @ dynamics_step_${step}"
        echo "==========================================================="
        python scripts/evaluate.py --config configs/evaluation.yaml -- \
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
            run_name="${mode}/dynamics_step_${step}"
    done
done

echo "All runs complete."
