"""
flavri.models.dmpnn
Pure PyTorch implementation of Directed Message Passing Neural Network (D-MPNN)
with Multi-Head Attentive Readout Pooling.
Scales to depth=6, hidden_dim=384.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from flavri.featurizer import BatchMolGraph, ATOM_FDIM, BOND_FDIM


class MultiHeadAttentivePooling(nn.Module):
    """
    Multi-Head Attentive Readout Pooling over atoms:
    K learned queries (default 4 heads).
    Each head computes attention scores over all atoms in the graph,
    extracting distinct structural motifs (e.g. heteroatom centers, aromatic rings, alkyl chains).
    Concatenates heads and projects to hidden_dim.
    """
    def __init__(self, hidden_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.w_att = nn.Linear(hidden_dim, hidden_dim)
        self.v_att = nn.Linear(hidden_dim, num_heads, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, atom_h: torch.Tensor, a2g: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        atom_h: (total_atoms, hidden_dim)
        a2g: (total_atoms,) graph indices in [0, batch_size - 1]
        batch_size: int
        Returns: (batch_size, hidden_dim)
        """
        # (total_atoms, num_heads)
        att_scores = self.v_att(torch.tanh(self.w_att(atom_h)))

        # Numerically stable segmented softmax per graph across each head
        max_scores = torch.full((batch_size, self.num_heads), -1e9, device=atom_h.device)
        max_scores.scatter_reduce_(0, a2g.unsqueeze(-1).expand_as(att_scores), att_scores, reduce="amax")

        exp_scores = torch.exp(att_scores - max_scores[a2g])

        sum_exp = torch.zeros((batch_size, self.num_heads), device=atom_h.device)
        sum_exp.scatter_add_(0, a2g.unsqueeze(-1).expand_as(exp_scores), exp_scores)

        weights = exp_scores / (sum_exp[a2g] + 1e-9)  # (total_atoms, num_heads)

        # For each head, compute weighted atom features
        head_outputs = []
        for k in range(self.num_heads):
            w_k = weights[:, k : k + 1]  # (total_atoms, 1)
            weighted_h = atom_h * w_k
            pooled_k = torch.zeros((batch_size, self.hidden_dim), device=atom_h.device)
            pooled_k.scatter_add_(0, a2g.unsqueeze(-1).expand_as(weighted_h), weighted_h)
            head_outputs.append(pooled_k)

        # Concatenate heads and project: (batch_size, hidden_dim * K) -> (batch_size, hidden_dim)
        multi_head = torch.cat(head_outputs, dim=-1)
        return self.proj(multi_head)


class DMPNNBackbone(nn.Module):
    """
    Directed Message Passing Neural Network (D-MPNN) Backbone (FLAVRI 2.0).
    Produces:
      - z: (batch_size, hidden_dim) pooled latent representation in R^512
      - atom_h: (total_atoms, hidden_dim) atom-level representations
    """
    def __init__(
        self,
        atom_fdim: int = ATOM_FDIM,
        bond_fdim: int = BOND_FDIM,
        hidden_dim: int = 512,
        depth: int = 6,
        dropout: float = 0.1,
        num_readout_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

        # Initial bond message projection: [x_u || e_uv] -> hidden_dim
        self.W_i = nn.Linear(atom_fdim + bond_fdim, hidden_dim, bias=False)

        # Directed message passing linear layer
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Atom representation projection: [x_v || sum m_uv] -> hidden_dim
        self.W_a = nn.Linear(atom_fdim + hidden_dim, hidden_dim)

        # Multi-head attentive readout
        self.readout = MultiHeadAttentivePooling(hidden_dim, num_heads=num_readout_heads)

    def forward(self, batch_graph: BatchMolGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs D-MPNN message passing on batch_graph.
        Returns:
            z: latent vectors in R^(batch_size, hidden_dim)
            atom_h: atom-level representations in R^(total_atoms, hidden_dim)
        """
        device = batch_graph.f_atoms.device

        # Special case: single atom molecules without any bonds
        if batch_graph.total_bonds == 0:
            atom_h = F.relu(self.W_a(torch.cat([
                batch_graph.f_atoms,
                torch.zeros((batch_graph.total_atoms, self.hidden_dim), device=device)
            ], dim=-1)))
            z = self.readout(atom_h, batch_graph.a2g, batch_graph.batch_size)
            return z, atom_h

        # Bond source atoms: batch_graph.b2u (1-indexed for bonds, index 0 is padding)
        u_atoms = batch_graph.b2u[1:]  # (total_bonds,)
        f_u_atoms = batch_graph.f_atoms[u_atoms]  # (total_bonds, atom_fdim)
        f_bonds = batch_graph.f_bonds[1:]         # (total_bonds, bond_fdim)

        init_input = torch.cat([f_u_atoms, f_bonds], dim=-1)  # (total_bonds, atom_fdim + bond_fdim)
        m_0 = F.relu(self.W_i(init_input))                     # (total_bonds, hidden_dim)

        # Prepend padding row 0
        zero_row = torch.zeros((1, self.hidden_dim), device=device)
        m = torch.cat([zero_row, m_0], dim=0)                  # (total_bonds + 1, hidden_dim)

        # Message passing for `depth` steps
        for _ in range(self.depth - 1):
            incoming_m = m[batch_graph.b2b]
            sum_incoming = incoming_m.sum(dim=1)
            m_update = F.relu(m_0 + self.W_m(sum_incoming[1:]))
            m_update = self.dropout(m_update)
            m = torch.cat([zero_row, m_update], dim=0)

        # Aggregate incoming bond messages to destination atoms v
        incoming_to_atoms = m[batch_graph.a2b]
        sum_m_v = incoming_to_atoms.sum(dim=1)

        # Atom representations: h_v = ReLU(W_a [x_v || sum m_uv])
        atom_input = torch.cat([batch_graph.f_atoms, sum_m_v], dim=-1)
        atom_h = F.relu(self.W_a(atom_input))
        atom_h = self.dropout(atom_h)

        # Multi-head attentive graph pooling readout -> (batch_size, hidden_dim)
        z = self.readout(atom_h, batch_graph.a2g, batch_graph.batch_size)
        return z, atom_h
