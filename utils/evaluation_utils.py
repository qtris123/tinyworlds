"""Rollout primitives for the Genie-style PSNR/Delta_t-PSNR evaluation pipeline.

These functions are pure (no global state, no IO, no logging) so the eval script can
call them in a tight loop. The autoregressive rollout below mirrors the loop in
`scripts/run_inference.py` (lines 101-136) but takes the action timeline as an explicit
input — so a single `rollout(...)` call evaluates either the GT-LAM-actions branch
(producing x_hat_t) or the random-actions branch (producing x_hat_prime_t) of the
Delta_t PSNR metric defined in genie.pdf section 3:

    Delta_t PSNR = PSNR(x_t, x_hat_t) - PSNR(x_t, x_hat_prime_t)

Conditioning convention (verified against `models/norms.py::AdaptiveNormalizer`):
when the conditioning sequence has length `T-1` for an input of length `T`, the
norm prepends a zero so that action `a_k` drives frame `z_{k+1}`. We therefore
build a per-clip `action_buffer` of length `T_total - 1` and slice
`action_buffer[:, i : i + context_window]` per rollout step.
"""

from typing import Optional

import torch


@torch.no_grad()
def tokenizer_reconstruct(video_tokenizer, frames: torch.Tensor) -> torch.Tensor:
    # frames: [B, T, C, H, W] in [-1, 1]
    # returns: [B, T, C, H, W] reconstruction through the FSQ-VAE
    embeddings = video_tokenizer.encoder(frames)  # [B, T, P, L]
    quantized = video_tokenizer.quantizer(embeddings)
    return video_tokenizer.decoder(quantized)  # [B, T, C, H, W]


@torch.no_grad()
def _make_action_buffer(
    latent_action_model,
    ground_truth_frames: torch.Tensor,
    n_actions: int,
    *,
    action_mode: str,
    action_seed: Optional[int],
) -> torch.Tensor:
    # ground_truth_frames: [B, T_total, C, H, W]
    # returns: [B, T_total - 1, A] action latents in [-1, 1]
    B, T_total = ground_truth_frames.shape[:2]

    if action_mode == 'gt_lam':
        return latent_action_model.encode(ground_truth_frames)  # [B, T_total - 1, A]

    if action_mode == 'random':
        if action_seed is None:
            raise ValueError("action_mode='random' requires action_seed to be set for determinism")
        # CPU generator -> portable across devices; move integer tensor to device after sampling
        cpu_gen = torch.Generator(device='cpu').manual_seed(int(action_seed))
        random_indices = torch.randint(
            low=0, high=n_actions, size=(B, T_total - 1), generator=cpu_gen,
        ).to(ground_truth_frames.device)
        return latent_action_model.quantizer.get_latents_from_indices(random_indices)  # [B, T_total - 1, A]

    raise ValueError(f"unknown action_mode: {action_mode!r}")


@torch.no_grad()
def rollout(
    video_tokenizer,
    latent_action_model,
    dynamics_model,
    ground_truth_frames: torch.Tensor,
    *,
    action_mode: str,
    n_actions: int,
    context_window: int,
    T_pred: int,
    prediction_horizon: int = 1,
    num_steps: int = 10,
    temperature: float = 0.0,
    action_seed: Optional[int] = None,
    teacher_forced: bool = False,
) -> torch.Tensor:
    """Autoregressively predict T_pred frames after the first context_window GT frames.

    ground_truth_frames: [B, context_window + T_pred, C, H, W] in [-1, 1].
        For action_mode='gt_lam', all T_total frames are passed through the LAM encoder
        once to derive the action timeline (i.e. these are the ground-truth-derived
        actions Genie calls a-tilde_{1:t}). For action_mode='random', only the shape and
        n_actions matter for sampling; the GT pixels are not consulted for the action
        timeline.

    When `teacher_forced=True`, the per-step context is taken from a sliding window
    over the GT frames (the model only ever sees real frames as context, and errors
    do not compound). When False (default, matching the Genie protocol), the context
    is built from the model's own previously generated frames.

    Returns: [B, T_pred, C, H, W] predicted frames in [-1, 1].
    """
    if prediction_horizon != 1:
        raise NotImplementedError(
            "eval rollout currently restricted to prediction_horizon=1 (autoregressive 1-frame "
            "predictions, matching Genie's t-by-t rollout)"
        )

    B, T_total = ground_truth_frames.shape[:2]
    expected_T = context_window + T_pred
    if T_total != expected_T:
        raise ValueError(
            f"ground_truth_frames must have T = context_window + T_pred = {expected_T}; got {T_total}"
        )

    action_buffer = _make_action_buffer(
        latent_action_model,
        ground_truth_frames,
        n_actions=n_actions,
        action_mode=action_mode,
        action_seed=action_seed,
    )  # [B, T_total - 1, A]

    generated = ground_truth_frames[:, :context_window].clone()  # [B, context_window, C, H, W]

    def idx_to_latents(idx):
        return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

    for i in range(T_pred):
        if teacher_forced:
            ctx = ground_truth_frames[:, i : i + context_window]  # [B, context_window, C, H, W]
        else:
            ctx = generated[:, -context_window:]  # [B, context_window, C, H, W]
        ctx_indices = video_tokenizer.tokenize(ctx)  # [B, context_window, P]
        ctx_latents = video_tokenizer.quantizer.get_latents_from_indices(ctx_indices)  # [B, context_window, P, L]

        # conditioning length must be context_window (= T-1 since prediction_horizon=1);
        # AdaptiveNormalizer will prepend a zero to align a_k with z_{k+1}
        cond = action_buffer[:, i : i + context_window]  # [B, context_window, A]

        next_latents = dynamics_model.forward_inference(
            context_latents=ctx_latents,
            prediction_horizon=prediction_horizon,
            num_steps=num_steps,
            index_to_latents_fn=idx_to_latents,
            conditioning=cond,
            temperature=temperature,
        )  # [B, context_window + prediction_horizon, P, L]

        next_frame = video_tokenizer.detokenize(next_latents[:, -prediction_horizon:])  # [B, 1, C, H, W]
        generated = torch.cat([generated, next_frame], dim=1)

    return generated[:, context_window:]  # [B, T_pred, C, H, W]
