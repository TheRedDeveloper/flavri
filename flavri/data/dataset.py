"""
flavri.data.dataset
PyTorch Dataset classes for Singles and Mixture Pairs.
Preloads SMILES strings and target/mask tensors into memory for fast iteration.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from flavri.constants import (
    TARGET_COLS_SINGLES,
    MASK_COLS_SINGLES,
    TARGET_COLS_PAIRS,
    MASK_COLS_PAIRS,
)


class FlavriSinglesDataset(Dataset):
    """
    Dataset for single molecules (singles_train.csv / singles_test.csv).
    """
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        df = pd.read_csv(self.csv_path)
        
        self.smiles = df["smiles"].values.tolist()
        self.targets = df[TARGET_COLS_SINGLES].values.astype(np.float32)
        self.masks = df[MASK_COLS_SINGLES].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "smiles": self.smiles[idx],
            "target": torch.from_numpy(self.targets[idx]),
            "mask": torch.from_numpy(self.masks[idx]),
        }


class FlavriPairsDataset(Dataset):
    """
    Dataset for pairwise mixtures (pair_train.csv / pair_test.csv).
    """
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        df = pd.read_csv(self.csv_path)

        self.smiles_a = df["smiles_a"].values.tolist()
        self.smiles_b = df["smiles_b"].values.tolist()
        
        self.deltas = df[TARGET_COLS_PAIRS].values.astype(np.float32)
        self.masks = df[MASK_COLS_PAIRS].values.astype(np.float32)
        
        self.has_cold_type = "cold_type" in df.columns
        self.cold_type = df["cold_type"].values.tolist() if self.has_cold_type else None

    def __len__(self) -> int:
        return len(self.smiles_a)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "smiles_a": self.smiles_a[idx],
            "smiles_b": self.smiles_b[idx],
            "delta": torch.from_numpy(self.deltas[idx]),
            "mask": torch.from_numpy(self.masks[idx]),
        }
        if self.has_cold_type:
            item["cold_type"] = self.cold_type[idx]
        return item
