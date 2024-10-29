# encdec_data.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple

class EHRDataset(Dataset):
    def __init__(
        self,
        static_data: np.ndarray,
        temporal_data: np.ndarray,
        mask_data: np.ndarray,
        time_data: np.ndarray
    ):
        self.static_data = torch.FloatTensor(static_data)
        self.temporal_data = torch.FloatTensor(temporal_data)
        self.mask_data = torch.FloatTensor(mask_data)
        self.time_data = torch.FloatTensor(time_data)
    
    def __len__(self):
        return len(self.static_data)
    
    def __getitem__(self, idx):
        return {
            'static': self.static_data[idx],
            'temporal': self.temporal_data[idx],
            'mask': self.mask_data[idx],
            'time': self.time_data[idx]
        }

def create_dataloaders(
    static_data: np.ndarray,
    temporal_data: np.ndarray,
    mask_data: np.ndarray,
    time_data: np.ndarray,
    batch_size: int,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Calculate split indices
    dataset_size = len(static_data)
    indices = np.random.permutation(dataset_size)
    split_idx = int(np.floor(val_split * dataset_size))
    train_indices = indices[split_idx:]
    val_indices = indices[:split_idx]
    
    # Create train dataset
    train_dataset = EHRDataset(
        static_data[train_indices],
        temporal_data[train_indices],
        mask_data[train_indices],
        time_data[train_indices]
    )
    
    # Create validation dataset
    val_dataset = EHRDataset(
        static_data[val_indices],
        temporal_data[val_indices],
        mask_data[val_indices],
        time_data[val_indices]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader