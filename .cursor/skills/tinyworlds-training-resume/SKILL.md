---
name: tinyworlds-training-resume
description: Diagnoses and fixes mid-run training resume on the TinyWorlds dynamics stage (and the sibling LAM / video-tokenizer trainers, which share the same checkpoint shape). Covers the latent "--checkpoint loads weights only" bug in `scripts/train_dynamics.py`, the FQN-keyed optim state_dict produced by `get_optimizer_state_dict`, the `LambdaLR.initial_lr` ordering trap, and the deterministic-DistributedSampler fast-forward trick. Use when continuing a partially-trained dynamics run after a node loss, server preemption, or planned restart — especially when the next person asks "why does loss spike for ~20 steps after I resume?", "why does my LR start from warmup again?", "how do I verify a resume change without GPUs?", or "is the dataloader bit-deterministic across restarts?".
---

# TinyWorlds — Training Resume

This skill documents how a `dynamics_step_N/` checkpoint actually gets turned back into a running training process, what the existing trainer code did NOT load before this patch, the fix that's now landed in `scripts/train_dynamics.py`, and a CPU-only smoke test the next agent can run to validate the resume path before booking a GPU.

All paths are relative to `tinyworlds/`.

## 1. The triggering situation (June 25, 2026)

The 4× GH200 server hosting `results/2026_06_24_15_31_15/` had to be released ~7000 steps into a 30000-step run. Latest healthy checkpoint at the time was `dynamics/checkpoints/dynamics_step_23000/`:

| File | Size | Contents |
|---|---|---|
| `model_state_dict.pt` | 305 MB | DDP-gathered full model weights |
| `optim_state_dict.pt` | 611 MB | AdamW `exp_avg`, `exp_avg_sq`, per-param `step` — saved via `get_optimizer_state_dict(full_state_dict=True)`, so keys are **FQN strings** (`'transformer.blocks.0.spatial_attn.q_proj.weight'`), not the integer positional keys that bare `optim.load_state_dict()` expects |
| `state.pt` | 3 KB | `scheduler_state_dict` (`last_epoch=23001`, `_last_lr=[1.5015e-4, …]`), `config`, `step=23000`, `timestamp` |

W&B run id was `0zdwzzbq` (`wandb/run-20260624_153124-0zdwzzbq`). Git HEAD at the time: `c6e57ed8edbddfa86e42a4f4b667c617a8556488`.

Everything needed for a true resume was already on disk. The trainer just wasn't loading it.

## 2. What the un-patched trainer did with `--checkpoint`

Before the patch, `args.checkpoint=…/dynamics_step_23000` triggered exactly one thing in `scripts/train_dynamics.py`:

```python
if args.checkpoint:
    dynamics_model, _ = load_dynamics_from_checkpoint(...)   # loads model weights only
```

Everything else was reset to a fresh-run state:

| Component | Saved? | Loaded by `--checkpoint`? | Symptom on resume |
|---|---|---|---|
| Model weights | yes | **yes** | — |
| AdamW `exp_avg`, `exp_avg_sq` | yes (`optim_state_dict.pt`) | **no** — fresh AdamW | β2-cold-start: for ~`1/(1-β2)≈20` steps with β2=0.95, `v_t` is artificially small → effective step `lr·m/√v` blows up → loss spike of the exact shape described in `tinyworlds-grad-norm-diagnosis` §4 |
| Scheduler `last_epoch` | yes (`state.pt['scheduler_state_dict']`) | **no** — fresh `LambdaLR` | LR re-warms from 0 over 1500 steps and decays from the wrong absolute position |
| Global step counter | yes (`state.pt['step']`) | **no** — `for i in range(0, n_updates)` | Loop runs 30000 *more* steps instead of `30000 − 23001` more |
| DataLoader / sampler position | **no** | n/a | First batch consumed is dataset index 0 again — but see §5, the sampler is deterministic so we can fast-forward by replaying `next()` calls |
| `torch` RNG (for `DynamicsModel` random masking) | **no** | n/a | Mask patterns differ from original run. Not catastrophic — model is trained to be mask-invariant. |

Only the first row is harmless; rows 2–4 will materially distort the remaining 7000 steps. Row 5 is the one we accept.

## 3. The fix that landed

Patch is in `scripts/train_dynamics.py`. Three edits:

1. **Imports** — added `from pathlib import Path` and `from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict, StateDictOptions`.

2. **Resume block** (immediately after the schedulers are constructed, ~line 131). When `args.checkpoint` is set, load:
   - `optim_state_dict.pt` via `set_optimizer_state_dict(model, optimizers[0], …, StateDictOptions(full_state_dict=True, broadcast_from_rank0=False))`. This handles the FQN→positional remap against the live (possibly DDP/FSDP-wrapped) model and works in both distributed and single-process cases. **Bare `optimizers[0].load_state_dict(opt_sd)` does NOT work** on this file because the saved keys are FQN strings, not integers.
   - `state_blob['scheduler_state_dict']` via `schedulers[0].load_state_dict(...)`. LambdaLR is a pure function of `last_epoch + base_lrs`, so this is bit-identical to what the original run would have applied next.
   - Optional `all_optimizers.pt` / `all_schedulers.pt` for the Muon+AdamW split-optimizer case (saved only on rank 0).
   - `start_step = int(state_blob['step']) + 1` (saved_step is the iteration that already had `opt.step()` and `sched.step()` applied before saving, so resume picks up at the next iteration).

3. **Two consumers of `start_step`:**
   - Before the main loop: a dataloader fast-forward (gated by `NG_RESUME_FAST_FORWARD=1`, default on) that drains `start_step * gradient_accumulation_steps` batches from `train_iter`. Safe to skip with `=0` if disk I/O is too slow (you'll replay ≤14% of the dataset over 7k more steps with effective batch 512).
   - Loop bounds: `for i in tqdm(range(start_step, args.n_updates), …)`.

### 3a. Trap: scheduler must be constructed BEFORE `set_optimizer_state_dict`

`LambdaLR.__init__` (via `LRScheduler.__init__`) writes `initial_lr` onto every `optimizer.param_groups[i]`. The saved `optim_state_dict.pt` also contains `param_groups.*.initial_lr`. If you attach the scheduler *after* `set_optimizer_state_dict`, the unflatten step fails with:

```
KeyError: 'param_groups.0.initial_lr'
```

…because the live optimizer doesn't yet have that key. The current code in `scripts/train_dynamics.py` already builds schedulers at line 128 and runs `set_optimizer_state_dict` at line 151, so this is correct. **Don't reorder.**

### 3b. Trap: don't bare-`load_state_dict` the optim file

```python
optimizers[0].load_state_dict(opt_sd)   # ❌ KeyError because keys are FQN strings
```

is wrong for any file produced by `get_optimizer_state_dict(... full_state_dict=True)`. Always go through `set_optimizer_state_dict` here. The non-distributed case works the same way — pass `broadcast_from_rank0=False`.

## 4. Verifying without GPUs

A CPU smoke test lives at `scripts/diag_resume_state_smoketest.py`. It exercises the same code path the trainer uses, against any `dynamics_step_*` directory.

```bash
cd tinyworlds
python scripts/diag_resume_state_smoketest.py \
  results/2026_06_24_15_31_15/dynamics/checkpoints/dynamics_step_23000
```

Expected output (all five lines present, then `ALL SMOKE CHECKS PASSED`):

```
[resume-smoke] model loaded
[resume-smoke] optimizer state loaded (581 param entries)
[resume-smoke] AdamW exp_avg_sq is warmed (non-zero) -> no cold-start  (largest tensor abs-mean = 2.811e-10)
[resume-smoke] scheduler last_epoch matches state.pt (23001); lr=1.501546e-04
[resume-smoke] start_step = saved_step + 1  (23000 → 23001)

ALL SMOKE CHECKS PASSED
```

What each line proves:

| Check | What it would catch |
|---|---|
| `optimizer state loaded (N param entries)` with `N>0` | The FQN→positional remap inside `set_optimizer_state_dict` worked; you didn't silently get an empty `optimizer.state` dict and a hidden cold start. |
| `AdamW exp_avg_sq is warmed (non-zero)` | The Adam second-moment EMA is restored, not zero. If this assert fires, **do not resume** — `optim_state_dict.pt` is the wrong file or was saved empty (compare against the `1315-byte empty-dict bug` documented in `tinyworlds-dynamics-training-observations` §2). |
| `scheduler last_epoch matches state.pt` + `lr=1.501546e-04` | The cosine scheduler is at the right position; the very next `sched.step()` will produce the LR the original run was about to use. |
| `start_step = saved_step + 1` | Loop bounds will be `range(23001, 30000)`, i.e. 6999 more iterations, not 30000. |

This smoke test runs on CPU in ~15 seconds on the current host. After running it once and seeing `ALL SMOKE CHECKS PASSED`, the resume code is end-to-end validated against the *real* on-disk artifacts. The remaining unknowns are GPU-only (DDP collective behavior under the new world_size, AMP autocast on the new GPU, dataset load speed).

This was the verification path used to land the patch on a CPU-only box. As of the patch landing, **no GPU verification has been done yet** — that has to happen on the new server.

## 5. Why the data fast-forward is bit-deterministic

`datasets/data_utils.py:153` builds the train sampler as:

```python
DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
```

The DistributedSampler's `__iter__` produces a `torch.randperm` seeded by `self.seed + self.epoch`. `self.seed` defaults to 0; `self.epoch` is only changed by `sampler.set_epoch(N)`, **which the dynamics trainer never calls** (this is also flagged as a separate issue in `tinyworlds-grad-norm-diagnosis` §3 "Spikes cluster at fixed step periods", because the same hard batch reappears every epoch). For resume purposes the absence of `set_epoch` is a **feature**: every fresh process produces the identical permutation, so draining `start_step * grad_accum` batches from a fresh iterator positions the next `next()` call at the same dataset index the original run was about to consume.

**Required invariants** for the fast-forward to be bit-identical:

- Same `world_size` (was 4 on the original box).
- Same `batch_size_per_gpu` (was 32).
- Same `gradient_accumulation_steps` (was 4).
- Same dataset bytes (`/data/zelda_frames.h5`).
- Same partition (`enable_partition=true`, `train_ratio=0.8`, `eval_ratio=0.1` — pinned in `results/2026_06_24_15_31_15/dynamics/data_partition.yaml`).

If any of those change, the per-rank index sequence changes and the math is wrong. In that case, set `NG_RESUME_FAST_FORWARD=0` and accept the replay.

## 6. The actual resume command for `dynamics_step_23000`

On the new server, after the bundle is unpacked under `tinyworlds/`:

```bash
cd tinyworlds
# (Always smoke-test first.)
python scripts/diag_resume_state_smoketest.py \
  results/2026_06_24_15_31_15/dynamics/checkpoints/dynamics_step_23000

# Resume — preserves the same W&B run; same world_size / batch_size / grad_accum
# as the original launch so the dataloader fast-forward is deterministic.
WANDB_RUN_ID=0zdwzzbq WANDB_RESUME=must \
EXTRA_OVERRIDES="checkpoint=results/2026_06_24_15_31_15/dynamics/checkpoints/dynamics_step_23000" \
  bash scripts/launch_dynamics_4gpu.sh
```

What you should see at startup, in rank-0 stdout:

```
[resume] checkpoint: results/2026_06_24_15_31_15/dynamics/checkpoints/dynamics_step_23000
[resume] saved_step=23000 → start_step=23001, lr=1.501e-04
[resume] n_updates=30000 (remaining=6999)
[resume] fast-forwarding dataloader by 92004 batches/rank
ff-loader: 100%|██████████| 92004/92004 [...]
```

Then the main tqdm should start at `23001/30000` and `train/loss` should resume **near where it left off** (around 0.14-ish given the trajectory in `tinyworlds-dynamics-training-observations` §1). If you instead see:

- `train/loss` jumping to 0.5–1.0 for ~20 steps before recovering → the optimizer-state load silently no-op'd (`optimizer.state` may be empty). Re-run the smoke test on the same checkpoint; if it now fails, this is the same `1315-byte empty-dict` bug from `tinyworlds-dynamics-training-observations` §2 — the on-disk `optim_state_dict.pt` is corrupted and you must fall back to an earlier step.
- LR starting at `~4e-5` and ramping → scheduler load no-op'd. Inspect `state.pt['scheduler_state_dict']['last_epoch']` directly; if it's `None` or `0`, the saving side regressed.
- tqdm starting at `0/30000` → `start_step` wasn't applied. Likely a stale copy of `train_dynamics.py` from before the patch.

## 7. State of this fix as of writing

- Patch in `scripts/train_dynamics.py`: **landed**.
- CPU smoke test (`scripts/diag_resume_state_smoketest.py`): **landed** and passes on the source server against `dynamics_step_23000`.
- GPU verification: **NOT done yet**. The original server lost GPU access before training could be relaunched. The next-server agent should treat §6 step 1 (CPU smoke) as a mandatory gate before launching multi-GPU, and §6 step 2 stdout as the first thing to eyeball after launch.
- The fix is intentionally scoped to `train_dynamics.py`. `train_latent_actions.py` and `train_video_tokenizer.py` use the same `save_training_state` shape and would benefit from the same resume block, but neither has the immediate continuation requirement and they are out of scope for this skill.

## 8. Cross-references

- `tinyworlds-dynamics-training-observations` — for the U-shape eval-loss interpretation, the `1315-byte empty-dict` save-race bug post-mortem (informs how to recognize a corrupted `model_state_dict.pt`/`optim_state_dict.pt`), and the GH200 memory budget.
- `tinyworlds-grad-norm-diagnosis` — for the Adam preconditioner decoupling mechanism behind the β2 cold-start spike, and for what `train/grad_norm` and `train/grad_skipped` are expected to look like in the steps immediately after a clean resume vs. a botched one.
- `scripts/diag_resume_smoketest.py` — older sibling of `diag_resume_state_smoketest.py`. It only validates the model-weight load path (the QK-Norm `strict=False` migration); it does NOT exercise optimizer/scheduler/step resume and so will pass even when the resume block is buggy. Use `diag_resume_state_smoketest.py` for the full-resume check.
