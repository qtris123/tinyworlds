from torch.utils.data import Dataset
import cv2
import h5py
import os
from tqdm import tqdm
import numpy as np
import torch
from typing import Optional, Tuple, Union

# TODO: Try pre-caching video tokens and have dataloader load video tokens instead of frames
class VideoHDF5Dataset(Dataset):
    def __init__(
        self,
        video_path: str,
        transform=None, # postnormalization
        save_path: Optional[str] = None,
        train: bool = True,
        disable_test_split: bool = True,
        num_frames: int = 4, # context length
        resize_to: Tuple[int, int] = (64, 64), 
        fps: int = 30,
        sequence_stride: Optional[int] = None, # default 60//fps
        fraction_of_dataset: float = 1.0, # fraction of valid starting indices to expose
        load_chunk_size: int = 1000, # chunk size when reading from HDF5
        load_start_index: int = 0, # skip initial frames when reading cached HDF5
        preload_ratio: Optional[float] = None, # if set, only load this ratio of cached frames
        preprocess_read_step: int = 1, # step to subsample raw video during preprocessing
        preprocess_slice: Optional[Tuple[Union[int, float], Union[int, float]]] = None, # optional slice applied to preprocessed frames; can be (start_idx, end_idx) ints or (start_ratio, end_ratio) floats in [0,1]
        # Three-way chronological partition (train / eval / test).
        # Only active when enable_partition=True; takes precedence over disable_test_split.
        # Partition boundaries are applied after all other preprocessing (preload_ratio, etc.)
        # so the raw-frame indices are deterministic and consistent across dataset instances.
        enable_partition: bool = False,
        partition: str = 'train',  # 'train' | 'eval' | 'test'
        train_ratio: float = 0.8,
        eval_ratio: float = 0.1,
    ) -> None:
        self.transform = transform
        self.train = train
        self.num_frames = num_frames
        self.fps = fps
        self.frame_skip = max(1, (sequence_stride if sequence_stride is not None else max(1, 60 // fps)))
        self.fraction_of_dataset = float(fraction_of_dataset)
        self.resize_to = resize_to

        if save_path and os.path.exists(save_path):
            with h5py.File(save_path, 'r') as h5_file:
                frames_dset = h5_file['frames']
                total = len(frames_dset)
                n_frames = int(total if preload_ratio is None else max(0, min(total, int(total * preload_ratio))))

                target_h, target_w = self.resize_to
                self.data = []
                for i in tqdm(range(load_start_index, n_frames, load_chunk_size), desc=f"Loading {n_frames} frames"):
                    chunk = frames_dset[i:min(i + load_chunk_size, n_frames)][:]
                    for frame in chunk:
                        # cached .h5 frames may be stored at a different resolution than
                        # the model expects; downscale/upscale to the target size on load.
                        if frame.shape[0] != target_h or frame.shape[1] != target_w:
                            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        self.data.append(frame)
                self.data = np.array(self.data)
        else:
            frames = self._preprocess_video(
                video_path=video_path,
                resize_to=resize_to,
                read_step=preprocess_read_step,
                slice_spec=preprocess_slice,
            )

            if save_path:
                print(f"Saving preprocessed frames to {save_path}")
                with h5py.File(save_path, 'w') as f:
                    f.create_dataset('frames', data=frames, compression='lzf')
                # Reload into memory to ensure consistent path
                with h5py.File(save_path, 'r') as h5_file:
                    frames = h5_file['frames'][:]

            self.data = frames

        # Store the total number of raw frames before any partition slicing so
        # callers can report the full dataset size in settings files.
        self.total_raw_frames = len(self.data)

        if enable_partition:
            # Three-way chronological split: first train_ratio → train,
            # next eval_ratio → eval, remainder → test.
            n = len(self.data)
            split1 = int(n * train_ratio)
            split2 = min(int(n * (train_ratio + eval_ratio)), n)
            if partition == 'train':
                self.data = self.data[:split1]
            elif partition == 'eval':
                self.data = self.data[split1:split2]
            else:  # 'test'
                self.data = self.data[split2:]
        elif not disable_test_split:
            split_idx = int(0.9 * len(self.data))
            self.data = self.data[:split_idx] if train else self.data[split_idx:]

    def _preprocess_video(
        self,
        video_path: str,
        resize_to: Tuple[int, int],
        read_step: int = 1,
        slice_spec: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    ) -> np.ndarray:
        print(f"Preprocessing video {video_path}")
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        step = max(1, int(read_step))
        for i in tqdm(range(0, total_frames, step), desc="Processing video frames"):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
            frames.append(frame)

        video.release()
        frames = np.array(frames)

        if slice_spec is not None and len(frames) > 0:
            start, end = slice_spec
            n = len(frames)
            if isinstance(start, float) or isinstance(end, float):
                # interpret as ratios
                s = 0 if start is None else int(n * max(0.0, min(1.0, float(start))))
                e = n if end is None else int(n * max(0.0, min(1.0, float(end))))
            else:
                s = 0 if start is None else int(start)
                e = n if end is None else int(end)
            s = max(0, min(n, s))
            e = max(s, min(n, e))
            frames = frames[s:e]

        return frames

    def __len__(self) -> int:
        max_valid_index = int((len(self.data) - (self.num_frames * self.frame_skip)) * self.fraction_of_dataset)
        return max(0, max_valid_index)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}")

        frame_sequence = self.data[index:index + (self.num_frames * self.frame_skip):self.frame_skip]
        if len(frame_sequence) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {len(frame_sequence)} frames")

        frame_sequence = frame_sequence.astype(np.float32) / 255.0

        if self.transform:
            transformed_frames = []
            for frame in frame_sequence:
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            frame_sequence = torch.stack(transformed_frames, dim=0)
        else:
            frame_sequence = torch.from_numpy(frame_sequence).permute(0, 3, 1, 2)

        return frame_sequence, 0

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close() 

# TODO: add more datasets
class PongDataset(VideoHDF5Dataset):
    def __init__(self, video_path, transform=None, save_path=None, train=True, num_frames=1, resolution=(64, 64), fps=30, preload_ratio=1, disable_test_split=True, **kwargs):
        super().__init__(
            video_path=video_path,
            transform=transform,
            save_path=save_path,
            train=train,
            disable_test_split=disable_test_split,
            num_frames=num_frames,
            resize_to=resolution,
            fps=fps,
            preload_ratio=preload_ratio,
            load_chunk_size=1000,
            load_start_index=0,
            preprocess_read_step=10,  # keep every 10th frame from raw
            preprocess_slice=None,
            **kwargs,
        )

class PolePositionDataset(VideoHDF5Dataset):
    def __init__(self, video_path, transform=None, save_path=None, train=True, num_frames=4, resolution=(64, 64), fps=30, preload_ratio=1, disable_test_split=True, **kwargs):
        super().__init__(
            video_path=video_path,
            transform=transform,
            save_path=save_path,
            train=train,
            disable_test_split=disable_test_split,
            num_frames=num_frames,
            resize_to=resolution,
            fps=fps,
            preload_ratio=preload_ratio,
            sequence_stride=None,
            load_chunk_size=1000,
            load_start_index=0,
            preprocess_read_step=1,
            preprocess_slice=(1/50, 1/4),
            **kwargs,
        )

class SonicDataset(VideoHDF5Dataset):
    def __init__(self, video_path, transform=None, save_path=None, train=True, num_frames=4, resolution=(128, 128), fps=15, preload_ratio=1, disable_test_split=True, **kwargs):
        super().__init__(
            video_path=video_path,
            transform=transform,
            save_path=save_path,
            train=train,
            disable_test_split=disable_test_split,
            num_frames=num_frames,
            resize_to=resolution,
            fps=fps,
            preload_ratio=preload_ratio,
            sequence_stride=None,
            load_chunk_size=1000,
            load_start_index=100,
            preprocess_read_step=1,
            preprocess_slice=None,
            **kwargs,
        )

class PicoDoomDataset(VideoHDF5Dataset):
    def __init__(self, video_path, transform=None, save_path=None, train=True, num_frames=4, resolution=(128, 128), fps=30, preload_ratio=0.3, disable_test_split=True, **kwargs):
        super().__init__(
            video_path=video_path,
            transform=transform,
            save_path=save_path,
            train=train,
            disable_test_split=disable_test_split,
            num_frames=num_frames,
            resize_to=resolution,
            fps=fps,
            preload_ratio=preload_ratio,
            sequence_stride=None,
            load_chunk_size=1000,
            load_start_index=300,
            preprocess_read_step=1,
            preprocess_slice=None,
            **kwargs,
        )

class ZeldaDataset(VideoHDF5Dataset):
    def __init__(self, video_path, transform=None, save_path=None, train=True, num_frames=4, resolution=(64, 64), fps=15, preload_ratio=0.2, disable_test_split=True, **kwargs):
        super().__init__(
            video_path=video_path,
            transform=transform,
            save_path=save_path,
            train=train,
            disable_test_split=disable_test_split,
            num_frames=num_frames,
            resize_to=resolution,
            fps=fps,
            preload_ratio=preload_ratio,
            sequence_stride=None,
            load_chunk_size=1000,
            load_start_index=1000,
            preprocess_read_step=1,
            preprocess_slice=None,
            **kwargs,
        )
