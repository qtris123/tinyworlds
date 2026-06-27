#!/usr/bin/env bash
# Unattended driver: runs the parallel eval sweep, then aggregates a combined
# summary at the end. Designed to be launched inside `tmux new-session -d`
# (or with `nohup setsid ... &`) so it survives the controlling terminal
# closing.
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

OUT_PREFIX="${OUT_PREFIX:-evaluation_results}"
export OUT_PREFIX
EVAL_STEPS_ENV="${EVAL_STEPS:-2000 10000 18000 24000}"
export EVAL_STEPS="${EVAL_STEPS_ENV}"
SUMMARY_TITLE="${SUMMARY_TITLE:-Combined evaluation sweep}"
export SUMMARY_TITLE

mkdir -p "${OUT_PREFIX}"

START_TS="$(date -Iseconds)"
echo "===== eval sweep started: ${START_TS} (out=${OUT_PREFIX}, steps=${EVAL_STEPS}) =====" 1>&2

bash scripts/run_eval_sweep_parallel.sh
SWEEP_STATUS=$?
END_TS="$(date -Iseconds)"

echo "===== eval sweep finished: ${END_TS}, status=${SWEEP_STATUS} =====" 1>&2

if [[ ${SWEEP_STATUS} -eq 0 ]]; then
    echo "Aggregating ${OUT_PREFIX}/combined_summary.md ..."
    python - <<PY
import json
import os

out_prefix = "${OUT_PREFIX}"
title = "${SUMMARY_TITLE}"
steps = tuple(int(s) for s in "${EVAL_STEPS}".split())
modes = ('teacher_forced', 'free_running')
rows = {}
for mode in modes:
    rows[mode] = {}
    for step in steps:
        p = f'{out_prefix}/{mode}/dynamics_step_{step}/ZELDA.json'
        if os.path.exists(p):
            rows[mode][step] = json.load(open(p))

ceiling = None
for mode in modes:
    for step in steps:
        if step in rows[mode]:
            ceiling = rows[mode][step]['tokenizer_recon_psnr_mean']
            break
    if ceiling is not None:
        break

lines = []
lines.append(f'# {title}')
lines.append('')
lines.append('ZELDA holdout, context_window=4, T_pred=8, GT-LAM-inferred action sequence (Genie protocol), 32 clips, 5 random-action seeds for the Δ4 baseline.')
lines.append('')
if ceiling is not None:
    lines.append(f'- Tokenizer reconstruction PSNR ceiling: **{ceiling:.3f} dB**')
lines.append('')
lines.append('## PSNR(x_t, x_hat_t) per generated step (GT-LAM actions)')
lines.append('')
lines.append('| Mode | Step | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 | Δ4 |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for mode in modes:
    for step in steps:
        if step not in rows[mode]:
            continue
        d = rows[mode][step]
        gt = d['psnr_gt_actions_per_t']
        delta = d['delta_psnr_at_t4']
        lines.append('| ' + mode.replace('_', ' ') + f' | {step} | '
                     + ' | '.join(f'{v:.2f}' for v in gt) + f' | {delta:+.3f} |')

lines.append('')
lines.append('## Headline @ t=4')
lines.append('')
lines.append('| Mode | Step | PSNR(x,x_hat) | PSNR(x,x_hat\') | Δ4 |')
lines.append('|---|---:|---:|---:|---:|')
for mode in modes:
    for step in steps:
        if step not in rows[mode]:
            continue
        d = rows[mode][step]
        gt4 = d['psnr_gt_actions_per_t'][3]
        rnd4 = d['psnr_random_actions_per_t'][3]
        delta = d['delta_psnr_at_t4']
        lines.append(f'| {mode.replace("_", " ")} | {step} | {gt4:.3f} | {rnd4:.3f} | {delta:+.3f} |')

summary_path = os.path.join(out_prefix, 'combined_summary.md')
with open(summary_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'Wrote {summary_path}')
PY
fi

echo "DONE @ ${END_TS} (status=${SWEEP_STATUS})" > "${OUT_PREFIX}/_sweep_done.flag"
echo "" 1>&2
echo "===== combined_summary.md written, all done =====" 1>&2
exit ${SWEEP_STATUS}
