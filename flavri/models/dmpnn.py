"""
flavri.models.dmpnn
Pure PyTorch implementation of Directed Message Passing Neural Network (D-MPNN)
with attentive graph pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flavri.featurizer import BatchMolGraph, ATOM_FDIM, BOND_FDIM


class AttentiveGraphPooling(nn.Module):
    """
    Attentive Graph Pooling Readout:
    alpha_v = softmax_over_atoms(w_att^T tanh(W_att h_v))
    z_i = sum_{v in G_i} alpha_v h_v
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.w_att = nn.Linear(hidden_dim, hidden_dim)
        self.v_att = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, atom_h: torch.Tensor, a2g: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        atom_h: (total_atoms, hidden_dim)
        a2g: (total_atoms,) graph indices in [0, batch_size - 1]
        batch_size: int
        Returns: (batch_size, hidden_dim)
        """
        # (total_atoms, 1)
        att_scores = self.v_att(torch.tanh(self.w_att(atom_h))).squeeze(-1)

        # Numerically stable segmented softmax
        # Find max per graph for stability
        max_scores = torch.full((batch_size,), -1e9, device=atom_h.device)
        max_scores.scatter_reduce_(0, a2g, att_scores, reduce="amax")
        
        # Exp of shifted scores
        exp_scores = torch.exp(att_scores - max_scores[a2g])
        
        # Denominator per graph
        sum_exp = torch.zeros(batch_size, device=atom_h.device)
        sum_exp.scatter_add_(0, a2g, exp_scores)
        
        # Softmax weights (total_atoms, 1)
        weights = (exp_scores / (sum_exp[a2g] + 1e-9)).unsqueeze(-1)
        
        # Weighted atom features
        weighted_h = atom_h * weights
        
        # Sum per graph: (batch_size, hidden_dim)
        pooled = torch.zeros((batch_size, atom_h.size(-1)), device=atom_h.device)
        pooled.scatter_add_(0, a2g.unsqueeze(-1).expand_as(weighted_h), weighted_h)
        
        return pooled


class DMPNNBackbone(nn.Module):
    """
    Directed Message Passing Neural Network (D-MPNN) Backbone.
    Produces latent chemosensory fingerprint z_i in R^256.
    """
    def __init__(
        self,
        atom_fdim: int = ATOM_FDIM,
        bond_fdim: int = BOND_FDIM,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
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
        
        # Attentive graph readout
        self.readout = AttentiveGraphPooling(hidden_dim)

    def forward(self, batch_graph: BatchMolGraph) -> torch.Tensor:
        """
        Runs D-MPNN message passing on batch_graph.
        Returns: latent vectors z in R^(batch_size, hidden_dim)
        """
        device = batch_graph.f_atoms.device
        
        # Special case: single atom molecules without any bonds
        if batch_graph.total_bonds == 0:
            atom_h = F.relu(self.W_a(torch.cat([
                batch_graph.f_atoms,
                torch.zeros((batch_graph.total_atoms, self.hidden_dim), device=device)
            ], dim=-1)))
            return self.readout(atom_h, batch_graph.a2g, batch_graph.batch_size)

        # Bond source atoms: batch_graph.b2u (1-indexed for bonds, index 0 is padding)
        # Bond initial input: concat atom feature of source atom u and bond feature
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
            # batch_graph.b2b is (total_bonds + 1, max_incoming_bonds)
            # Gather incoming messages: (total_bonds + 1, max_incoming, hidden_dim)
            incoming_m = m[batch_graph.b2b]
            # Sum incoming messages along neighbors dimension: (total_bonds + 1, hidden_dim)
            sum_incoming = incoming_m.sum(dim=1)
            
            # Update: m^(t) = ReLU(m^(0) + W_m sum m^(t-1))
            # Only update for bonds 1..total_bonds
            m_update = F.relu(m_0 + self.W_m(sum_incoming[1:]))
            m_update = self.dropout(m_update)
            m = torch.cat([zero_row, m_update], dim=0)

        # Aggregate incoming bond messages to destination atoms v
        # batch_graph.a2b is (total_atoms, max_incoming_atoms)
        incoming_to_atoms = m[batch_graph.a2b]  # (total_atoms, max_incoming, hidden_dim)
        sum_m_v = incoming_to_atoms.sum(dim=1)  # (total_atoms, hidden_dim)

        # Atom representations: h_v = ReLU(W_a [x_v || sum m_uv])
        atom_input = torch.cat([batch_graph.f_atoms, sum_m_v], dim=-1)
        atom_h = F.relu(self.W_a(atom_input))
        atom_h = self.dropout(atom_h)

        # Attentive graph pooling readout -> (batch_size, hidden_dim)
        z = self.readout(atom_h, batch_graph.a2g, batch_graph.batch_size)
        return z
