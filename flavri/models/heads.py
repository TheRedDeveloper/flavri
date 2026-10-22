"""
flavri.models.heads
Sensory prediction heads:
1. SingleMoleculeHead (148 Channels)
2. CommutativeSynergyHead (148 Channels, symmetric & identity constrained)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flavri.constants import NUM_CHANNELS


class SingleMoleculeHead(nn.Module):
    """
    Single Molecule Sensory Profile Head:
    V_i = sigma( Linear(512 -> 148) ( GELU( Linear(512 -> 512) z_i ) ) )
    Output in [0, 1]^148.
    """
    def __init__(self, hidden_dim: int = 512, num_channels: int = NUM_CHANNELS, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_channels)

    def forward(self, z: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        z: (B, hidden_dim)
        Returns: (B, num_channels) probabilities in [0, 1] (or raw logits if return_logits=True)
        """
        h = self.dropout(self.act(self.fc1(z)))
        logits = self.fc2(h)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


class CommutativeSynergyHead(nn.Module):
    """
    Symmetric Commutative Interaction Synergy Head:
    Given z_A, z_B in R^512:
        u = z_A * z_B       (Co-activation / shared mechanism)
        v = |z_A - z_B|     (Divergence / contrast)
        [u || v] in R^1024
    Delta_AB = tanh( Linear(512 -> 148) ( GELU( Linear(1024 -> 512) [u || v] ) ) )
    Output in [-1, +1]^148.

    Guarantees:
    1. Exact Commutativity: Delta_AB == Delta_BA by construction.
    2. Identity neutrality: When A == B, v = 0 and final layer bias initialized to 0.
    """
    def __init__(self, hidden_dim: int = 512, num_channels: int = NUM_CHANNELS, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_channels)

        # Enforce zero bias and small weights initialization on the output layer
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        z_a: (B, hidden_dim)
        z_b: (B, hidden_dim)
        Returns: (B, num_channels) synergy deltas in [-1, +1]
        """
        u = z_a * z_b
        v = torch.abs(z_a - z_b)
        uv = torch.cat([u, v], dim=-1)
        h = self.dropout(self.act(self.fc1(uv)))
        delta = torch.tanh(self.fc2(h))
        return delta

    def compute_identity_penalty(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes identity self-interaction penalty: ||Delta_AA||_2^2
        z: (B, hidden_dim)
        """
        delta_aa = self.forward(z, z)
        return torch.mean(torch.sum(delta_aa ** 2, dim=-1))
