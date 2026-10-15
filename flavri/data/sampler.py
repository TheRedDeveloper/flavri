"""
flavri.data.sampler
Two-Tier Stratified Diversity Sampler:
Tier 1: Modality Balance (Exact 50/50 Taste vs. Odor molecules per batch).
Tier 2: Murcko Scaffold Diversity (Samples across distinct scaffold clusters,
        preventing single-family gradient saturation and batch redundancy).
"""

from typing import Iterator, List, Dict
import random
import numpy as np
import pandas as pd
from torch.utils.data import Sampler
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from flavri.constants import NUM_TASTE, MASK_COLS_SINGLES


def extract_scaffold(smiles: str) -> str:
    """Extracts canonical generic Murcko scaffold for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaffold:
                return scaffold
    except Exception:
        pass
    return "acyclic_or_linear"


class ScaffoldDiverseModalitySampler(Sampler[List[int]]):
    """
    Two-Tier Maximum-Entropy Batch Sampler:
    For every batch of size B:
      - B // 2 molecules from the Odor-active cohort, drawn across diverse scaffolds.
      - B // 2 molecules from the Taste-active cohort, drawn across diverse scaffolds.
    Guarantees:
      1. Zero-gradient starvation on odor or taste channels is mathematically impossible.
      2. No single chemical family (e.g. simple benzenes, aliphatic chains) dominates a batch.
    """
    def __init__(
        self,
        csv_path: str,
        batch_size: int = 128,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.half_b = batch_size // 2
        self.shuffle = shuffle
        self.seed = seed
        self.rng = random.Random(seed)

        df = pd.read_csv(csv_path)
        t_masks = df[MASK_COLS_SINGLES[:NUM_TASTE]].values
        o_masks = df[MASK_COLS_SINGLES[NUM_TASTE:]].values

        is_taste = t_masks.sum(axis=1) > 0
        is_odor = o_masks.sum(axis=1) > 0

        self.taste_indices = np.where(is_taste)[0].tolist()
        self.odor_indices = np.where(is_odor)[0].tolist()

        # Build scaffold clusters for odor cohort
        self.odor_scaffold_to_indices: Dict[str, List[int]] = {}
        for idx in self.odor_indices:
            scaf = extract_scaffold(df["smiles"].iloc[idx])
            self.odor_scaffold_to_indices.setdefault(scaf, []).append(idx)

        # Build scaffold clusters for taste cohort
        self.taste_scaffold_to_indices: Dict[str, List[int]] = {}
        for idx in self.taste_indices:
            scaf = extract_scaffold(df["smiles"].iloc[idx])
            self.taste_scaffold_to_indices.setdefault(scaf, []).append(idx)

        self.odor_scaffolds = list(self.odor_scaffold_to_indices.keys())
        self.taste_scaffolds = list(self.taste_scaffold_to_indices.keys())

        # Total number of batches defined by the larger cohort to ensure full coverage
        n_odor_batches = len(self.odor_indices) // self.half_b
        n_taste_batches = len(self.taste_indices) // self.half_b
        self.n_batches = max(n_odor_batches, n_taste_batches)

    def _sample_diverse(
        self,
        scaffolds: List[str],
        scaffold_map: Dict[str, List[int]],
        count: int,
    ) -> List[int]:
        """Samples `count` molecules drawn round-robin across distinct scaffolds."""
        if self.shuffle:
            self.rng.shuffle(scaffolds)

        selected: List[int] = []
        scaf_idx = 0
        while len(selected) < count:
            scaf = scaffolds[scaf_idx % len(scaffolds)]
            candidates = scaffold_map[scaf]
            selected.append(self.rng.choice(candidates))
            scaf_idx += 1

        return selected

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_batches):
            odor_batch = self._sample_diverse(
                self.odor_scaffolds,
                self.odor_scaffold_to_indices,
                self.half_b,
            )
            taste_batch = self._sample_diverse(
                self.taste_scaffolds,
                self.taste_scaffold_to_indices,
                self.half_b,
            )
            combined = odor_batch + taste_batch
            if self.shuffle:
                self.rng.shuffle(combined)
            yield combined

    def __len__(self) -> int:
        return self.n_batches
