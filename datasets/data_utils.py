import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import time
import os
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datasets.datasets import PongDataset, SonicDataset, PolePositionDataset, PicoDoomDataset, ZeldaDataset

DEFAULT_NUM_WORKERS = 8
DEFAULT_PREFETCH_FACTOR = 4
DEFAULT_PIN_MEMORY = True
DEFAULT_PERSISTENT_WORKERS = True


def _default_video_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def _load_video_dataset_pair(
    dataset_cls,
    video_rel_path,
    h5_rel_path,
    num_frames,
    transform=None,
    fps=30,
    preload_ratio=1,
    disable_test_split: bool = True,
    resize_to: Optional[Tuple[int, int]] = None,
    **kwargs,
):
    current_folder_path = os.getcwd()
    video_path = current_folder_path + video_rel_path
    preprocessed_path = current_folder_path + h5_rel_path
    transform = _default_video_transform() if transform is None else transform

    # Only override the dataset class's `resolution` default if the caller explicitly
    # asked to. This keeps existing training/inference call sites on their dataset
    # defaults (e.g. SONIC=128x128) while letting evaluate.py force a uniform 64x64.
    extra: dict = {}
    if resize_to is not None:
        extra['resolution'] = tuple(resize_to)

    train = dataset_cls(
        video_path,
        transform=transform,
        save_path=preprocessed_path,
        train=True,
        disable_test_split=disable_test_split,
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        **extra,
        **kwargs,
    )
    val = dataset_cls(
        video_path,
        transform=transform,
        save_path=preprocessed_path,
        train=False,
        disable_test_split=disable_test_split,
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        **extra,
        **kwargs,
    )
    return train, val


def load_pong(num_frames=1, fps=15, preload_ratio=1, disable_test_split: bool = True, resize_to: Optional[Tuple[int, int]] = None, **kwargs):
    return _load_video_dataset_pair(
        PongDataset,
        '/data/pong.mp4',
        '/data/pong_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        disable_test_split=disable_test_split,
        resize_to=resize_to,
        **kwargs,
    )


def load_sonic(num_frames=4, fps=15, preload_ratio=1, disable_test_split: bool = True, resize_to: Optional[Tuple[int, int]] = None, **kwargs):
    return _load_video_dataset_pair(
        SonicDataset,
        '/data/sonic_frames.mp4',
        '/data/sonic_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        disable_test_split=disable_test_split,
        resize_to=resize_to,
        **kwargs,
    )


def load_pole_position(num_frames=4, fps=15, preload_ratio=1, disable_test_split: bool = True, resize_to: Optional[Tuple[int, int]] = None, **kwargs):
    return _load_video_dataset_pair(
        PolePositionDataset,
        '/data/pole_position.mp4',
        '/data/pole_position_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        disable_test_split=disable_test_split,
        resize_to=resize_to,
        **kwargs,
    )


def load_picodoom(num_frames=4, fps=30, preload_ratio=1, disable_test_split: bool = True, resize_to: Optional[Tuple[int, int]] = None, **kwargs):
    return _load_video_dataset_pair(
        PicoDoomDataset,
        '/data/picodoom cleaned.mp4',
        '/data/picodoom_frames.h5',
        num_frames=num_frames,
        fps=30,
        preload_ratio=preload_ratio,
        disable_test_split=disable_test_split,
        resize_to=resize_to,
        **kwargs,
    )


def load_zelda(num_frames=4, fps=15, preload_ratio=1, disable_test_split: bool = True, resize_to: Optional[Tuple[int, int]] = None, **kwargs):
    return _load_video_dataset_pair(
        ZeldaDataset,
        '/data/Zelda oot2d 1 Cut.mp4',
        '/data/zelda_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        disable_test_split=disable_test_split,
        resize_to=resize_to,
        **kwargs,
    )


def data_loaders(train_data, val_data, batch_size, distributed=False, rank=0, world_size=1):
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False if train_sampler is not None else True,
        sampler=train_sampler,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=DEFAULT_PERSISTENT_WORKERS,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        drop_last=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False if val_sampler is not None else True,
        sampler=val_sampler,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=DEFAULT_PERSISTENT_WORKERS,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        drop_last=True
    )
    return train_loader, val_loader


def load_data_and_data_loaders(
    dataset,
    batch_size,
    num_frames=1,
    distributed=False,
    rank=0,
    world_size=1,
    fps=15,
    preload_ratio=1,
    disable_test_split: bool = True,
    resize_to: Optional[Tuple[int, int]] = None,
    # Partition params forwarded to VideoHDF5Dataset (used when enable_eval_loss=True)
    enable_partition: bool = False,
    partition: str = 'train',
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
):
    common = dict(
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        disable_test_split=disable_test_split,
        resize_to=resize_to,
        enable_partition=enable_partition,
        partition=partition,
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
    )
    if dataset == 'PONG':
        training_data, validation_data = load_pong(**common)
    elif dataset == 'SONIC':
        training_data, validation_data = load_sonic(**common)
    elif dataset == 'POLE_POSITION':
        training_data, validation_data = load_pole_position(**common)
    elif dataset == 'PICODOOM':
        training_data, validation_data = load_picodoom(**common)
    elif dataset == 'ZELDA':
        training_data, validation_data = load_zelda(**common)
    else:
        raise ValueError('Invalid dataset')

    training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size,
        distributed=distributed, rank=rank, world_size=world_size
    )
    x_train_var = np.var(training_data.data)

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def load_eval_data_and_loader(
    dataset,
    batch_size,
    num_frames=1,
    fps=15,
    preload_ratio=1,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    resize_to: Optional[Tuple[int, int]] = None,
):
    """Load the eval partition for in-training eval-loss monitoring.

    The eval partition is the chronological slice
    [train_ratio, train_ratio + eval_ratio) of the raw frames.  The DataLoader
    is intentionally non-distributed — all ranks compute eval loss on the same
    full eval set independently, and only rank-0 logs the result.

    Returns (eval_data, eval_loader).
    """
    common = dict(
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        disable_test_split=True,
        resize_to=resize_to,
        enable_partition=True,
        partition='eval',
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
    )
    if dataset == 'PONG':
        eval_data, _ = load_pong(**common)
    elif dataset == 'SONIC':
        eval_data, _ = load_sonic(**common)
    elif dataset == 'POLE_POSITION':
        eval_data, _ = load_pole_position(**common)
    elif dataset == 'PICODOOM':
        eval_data, _ = load_picodoom(**common)
    elif dataset == 'ZELDA':
        eval_data, _ = load_zelda(**common)
    else:
        raise ValueError('Invalid dataset')

    eval_loader = DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=DEFAULT_PERSISTENT_WORKERS,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        drop_last=False,
    )
    return eval_data, eval_loader


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def visualize_reconstruction(original, reconstruction, save_path=None, actions=None):
    # original: (B, C, H, W) or (B, T, C, H, W)
    # reconstruction: (B, C, H, W) or (B, T, C, H, W)
    # actions: optional (B, T-1) integer action indices, used to annotate transitions

    # move tensors to CPU and convert to float32 for matplotlib compatibility
    original = original.detach().to('cpu', dtype=torch.float32)
    reconstruction = reconstruction.detach().to('cpu', dtype=torch.float32)

    # handle single frames by expanding to sequences
    if original.dim() == 4:  # (B, C, H, W)
        original = original.unsqueeze(1)  # Add sequence dimension
    if reconstruction.dim() == 4:  # (B, C, H, W)
        reconstruction = reconstruction.unsqueeze(1)  # Add sequence dimension

    # take first 4 sequences, each of length 4 (or available length)
    num_sequences = min(4, original.shape[0])
    seq_length = min(4, original.shape[1])

    original = original[:num_sequences, :seq_length]  # (B, T, C, H, W)
    reconstruction = reconstruction[:num_sequences, :seq_length]  # (B, T, C, H, W)

    show_actions = actions is not None
    if show_actions:
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.35, wspace=0.15)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # original sequences
    orig_flat = original.reshape(-1, *original.shape[2:])  # (B*T, C, H, W)
    grid_orig = make_grid(orig_flat, nrow=seq_length, normalize=True, padding=2).clamp(0, 1)
    ax1.imshow(grid_orig.permute(1, 2, 0).contiguous().numpy())
    ax1.axis('off')
    ax1.set_title(f'Original Sequences ({num_sequences} sequences × {seq_length} frames)')

    # reconstructed sequences
    recon_flat = reconstruction.reshape(-1, *reconstruction.shape[2:])  # (B*T, C, H, W)
    grid_recon = make_grid(recon_flat, nrow=seq_length, normalize=True, padding=2).clamp(0, 1)
    ax2.imshow(grid_recon.permute(1, 2, 0).contiguous().numpy())
    ax2.axis('off')
    recon_title = f'Reconstructed Sequences ({num_sequences} sequences × {seq_length} frames)'
    if show_actions:
        recon_title += '\nframe[t+1] = dynamics(latent_code[t], action[t→t+1])'
    ax2.set_title(recon_title)

    if show_actions:
        # actions: (B, T-1) integer indices
        actions = actions.detach().cpu().long()
        n_transitions = min(seq_length - 1, actions.shape[1])
        action_grid = actions[:num_sequences, :n_transitions]  # (num_seq, n_transitions)

        # determine colour scale from the config-level n_actions if possible; fall back to data max
        n_actions_total = max(action_grid.max().item() + 1, 8)

        im = ax3.imshow(
            action_grid.float().numpy(),
            aspect='auto',
            cmap='tab10',
            vmin=0,
            vmax=n_actions_total - 1,
            interpolation='nearest',
        )

        for row in range(num_sequences):
            for col in range(n_transitions):
                idx_val = action_grid[row, col].item()
                ax3.text(
                    col, row, f'a={idx_val}',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white',
                )

        ax3.set_xticks(range(n_transitions))
        ax3.set_xticklabels([f't{t}→t{t+1}' for t in range(n_transitions)], fontsize=10)
        ax3.set_yticks(range(num_sequences))
        ax3.set_yticklabels([f'seq {s}' for s in range(num_sequences)], fontsize=10)
        ax3.set_title(
            'Action Codes Used for Reconstruction  '
            '(each predicted frame[t+1] requires latent_video_code[t] + action_code[t→t+1])',
            fontsize=10,
        )
        plt.colorbar(im, ax=ax3, label='Action index', fraction=0.02, pad=0.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
