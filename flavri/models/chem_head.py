"""
flavri.models.chem_head
Persistent Chemical Reconstruction Head for self-supervised pre-training
and auxiliary task regularization.
Predicts masked atom features (dimension ATOM_FDIM = 133) from atom embeddings.
"""

import torch
import torch.nn as nn
from flavri.featurizer import ATOM_FDIM


class ChemicalReconstructionHead(nn.Module):
    """
    Predicts ground-truth atom feature vectors (133 dims) from atom latent representations.
    Kept active during Stages 0, 1, and 2 as a chemical anchor to prevent representation drift.
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        atom_fdim: int = ATOM_FDIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, atom_fdim),
        )

    def forward(self, atom_h: torch.Tensor) -> torch.Tensor:
        """
        atom_h: (total_atoms, hidden_dim) atom embeddings from D-MPNN
        Returns: (total_atoms, atom_fdim) predicted atom features
        """
        return self.net(atom_h)
