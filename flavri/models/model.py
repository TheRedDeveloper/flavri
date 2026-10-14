"""
flavri.models.model
FLAVRI 2.0 Unified Multimodal Chemosensory Architecture:
- Backbone: Scaled 6-Layer D-MPNN (384d) with Multi-Head Attentive Readout
- Hybrid Feature Conditioning: Morgan / ECFP4 Fingerprint Projection (512d -> 128d -> 384d)
- Auxiliary Task Head: Persistent Chemical Reconstruction Head (Masked Atom/Bond features)
- Task Heads: SingleMoleculeHead (148 Channels) & CommutativeSynergyHead (Pairwise Interaction)
"""

from typing import Tuple, Optional, Any
import torch
import torch.nn as nn
from flavri.constants import NUM_CHANNELS
from flavri.featurizer import BatchMolGraph
from flavri.models.dmpnn import DMPNNBackbone
from flavri.models.heads import SingleMoleculeHead, CommutativeSynergyHead
from flavri.models.chem_head import ChemicalReconstructionHead


class FlavriMultimodalModel(nn.Module):
    """
    Decoupled Multimodal Chemosensory Architecture (FLAVRI 2.0):
    - D-MPNN Backbone (384d, depth 6) + Multi-Head Attentive Readout (4 heads)
    - Persistent Chemical Reconstruction Head (133-dim atom features)
    - SingleMoleculeHead (148 channels)
    - CommutativeSynergyHead (exact symmetry & identity neutrality)
    """
    def __init__(
        self,
        hidden_dim: int = 384,
        depth: int = 6,
        num_channels: int = NUM_CHANNELS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels

        self.backbone = DMPNNBackbone(
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            num_readout_heads=4,
        )
        self.chem_head = ChemicalReconstructionHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.single_head = SingleMoleculeHead(
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            dropout=dropout,
        )
        self.synergy_head = CommutativeSynergyHead(
            hidden_dim=hidden_dim,
            num_channels=num_channels,
            dropout=dropout,
        )

        # Hybrid Morgan / ECFP4 Fingerprint Projector & Fusion
        self.fp_dim = 512
        self.fp_proj = nn.Sequential(
            nn.Linear(self.fp_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def encode(self, batch_graph: BatchMolGraph, return_atom_h: bool = False) -> Any:
        """
        Encodes graph batch to latent vectors z in R^(B, hidden_dim).
        Optionally returns per-atom representations atom_h in R^(total_atoms, hidden_dim).
        """
        z_graph, atom_h = self.backbone(batch_graph)
        if hasattr(batch_graph, "fps") and batch_graph.fps is not None:
            z_fp = self.fp_proj(batch_graph.fps)
            z = self.fusion(torch.cat([z_graph, z_fp], dim=-1))
        else:
            z = z_graph

        if return_atom_h:
            return z, atom_h
        return z

    def predict_singles(
        self,
        batch_graph: BatchMolGraph,
        return_atom_h: bool = False,
    ) -> Any:
        """
        Predicts sensory profiles for single molecules.
        Returns:
            z: (B, hidden_dim) latent vectors
            v: (B, 148) single molecule sensory profiles in [0, 1]
            atom_h: (total_atoms, hidden_dim) if return_atom_h=True
        """
        z, atom_h = self.encode(batch_graph, return_atom_h=True)
        v = self.single_head(z)
        if return_atom_h:
            return z, v, atom_h
        return z, v

    def predict_chemistry(self, atom_h: torch.Tensor) -> torch.Tensor:
        """Predicts ground truth atom features from atom representations."""
        return self.chem_head(atom_h)

    def predict_synergy(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Computes commutative pairwise synergy deltas.
        Returns:
            delta: (B, 148) interaction synergy deltas in [-1, +1]
        """
        return self.synergy_head(z_a, z_b)

    def blend_profiles(
        self,
        v_a: torch.Tensor,
        v_b: torch.Tensor,
        delta_ab: torch.Tensor,
        w_a: torch.Tensor,
        w_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes composite blend profile under the FLAVRI Mixture Blending Law:
        y_blend = clamp( w_A * V_A + w_B * V_B + 4 * w_A * w_B * Delta_AB, 0.0, 1.0 )
        """
        if w_a.dim() == 1:
            w_a = w_a.unsqueeze(1)
        if w_b.dim() == 1:
            w_b = w_b.unsqueeze(1)

        linear_mix = w_a * v_a + w_b * v_b
        synergy_term = 4.0 * w_a * w_b * delta_ab
        y_blend = torch.clamp(linear_mix + synergy_term, min=0.0, max=1.0)
        return y_blend

    def predict_pairs_batch(
        self,
        unique_graphs: BatchMolGraph,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        w_a: Optional[torch.Tensor] = None,
        w_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        O(1) Graph Inference for a batch of mixture pairs:
        1. Encodes unique molecules once.
        2. Indexes z_A and z_B.
        3. Computes V_A, V_B, Delta_AB, and y_blend.
        """
        z_unique = self.encode(unique_graphs)  # (N_unique, hidden_dim)
        v_unique = self.single_head(z_unique)  # (N_unique, 148)

        z_a = z_unique[idx_a]
        z_b = z_unique[idx_b]
        v_a = v_unique[idx_a]
        v_b = v_unique[idx_b]

        delta_ab = self.predict_synergy(z_a, z_b)

        if w_a is None:
            w_a = torch.full((len(idx_a), 1), 0.5, device=idx_a.device)
        if w_b is None:
            w_b = torch.full((len(idx_b), 1), 0.5, device=idx_b.device)

        y_blend = self.blend_profiles(v_a, v_b, delta_ab, w_a, w_b)

        return delta_ab, y_blend, v_a, v_b
