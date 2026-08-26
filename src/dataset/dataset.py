"""
Multi-shard memory-mapped dataset loader for streaming multi-billion token datasets.
"""

import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


class MultiShardDataset(Dataset):
    """
    Seamlessly streams across multiple binary token shards (e.g. shard_0000.bin -> shard_0023.bin).
    O(1) RAM footprint by mapping individual shards on-demand.
    """
    def __init__(self, shards_dir: str, seq_len: int = 2048, dtype=np.uint16):
        self.shards_dir = shards_dir
        self.seq_len = seq_len
        self.dtype = dtype
        
        self.shard_files = sorted(glob.glob(os.path.join(shards_dir, "takatsuki_shard_*.bin")))
        if not self.shard_files:
            # Fallback to single file if shards directory doesn't have multiple shards
            single_file = os.path.join(shards_dir, "train.bin")
            if os.path.exists(single_file):
                self.shard_files = [single_file]
            else:
                raise FileNotFoundError(f"No binary token shards found in {shards_dir}")

        self.shard_lengths = []
        self.cumulative_samples = [0]
        
        total_tokens = 0
        for f in self.shard_files:
            file_size_bytes = os.path.getsize(f)
            tokens_in_shard = file_size_bytes // np.dtype(self.dtype).itemsize
            samples_in_shard = max(0, (tokens_in_shard - 1) // self.seq_len)
            
            self.shard_lengths.append(samples_in_shard)
            self.cumulative_samples.append(self.cumulative_samples[-1] + samples_in_shard)
            total_tokens += tokens_in_shard

        self.total_samples = self.cumulative_samples[-1]
        self.total_tokens = total_tokens
        print(f"Loaded MultiShardDataset: {len(self.shard_files)} shards | {self.total_tokens:,} tokens | {self.total_samples:,} sequences (seq_len={self.seq_len})")

        self._active_shard_idx = -1
        self._active_memmap = None

    def __len__(self) -> int:
        return self.total_samples

    def _get_memmap(self, shard_idx: int):
        if self._active_shard_idx != shard_idx:
            self._active_shard_idx = shard_idx
            self._active_memmap = np.memmap(self.shard_files[shard_idx], dtype=self.dtype, mode="r")
        return self._active_memmap

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Sample index {idx} out of range [0, {self.total_samples})")

        # Binary search for the right shard
        shard_idx = 0
        for i in range(len(self.cumulative_samples) - 1):
            if self.cumulative_samples[i] <= idx < self.cumulative_samples[i + 1]:
                shard_idx = i
                break

        local_idx = idx - self.cumulative_samples[shard_idx]
        data = self._get_memmap(shard_idx)

        start_idx = local_idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        
        chunk = torch.from_numpy(data[start_idx:end_idx].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_multi_shard_dataloader(
    shards_dir: str,
    batch_size: int = 8,
    seq_len: int = 2048,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    dataset = MultiShardDataset(shards_dir=shards_dir, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
