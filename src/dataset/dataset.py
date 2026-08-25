"""
Memory-mapped dataset reader and streaming token iterator for efficient SLM pre-training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class BinaryPretrainingDataset(Dataset):
    """
    Reads large memory-mapped binary files containing int32/uint16 token IDs.
    Extremely memory efficient (O(1) RAM usage regardless of dataset size).
    """
    def __init__(self, bin_path: str, seq_len: int = 2048, dtype=np.uint16):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.dtype = dtype
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary dataset not found at {bin_path}")
            
        self.data = np.memmap(bin_path, dtype=self.dtype, mode="r")
        self.total_tokens = len(self.data)
        self.n_samples = (self.total_tokens - 1) // self.seq_len

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        
        chunk = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_dataloader(
    bin_path: str,
    batch_size: int = 8,
    seq_len: int = 2048,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    dataset = BinaryPretrainingDataset(bin_path=bin_path, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
