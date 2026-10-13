"""
flavri.data.collator
Collate functions for batching molecular graphs.
Implements O(1) graph batch deduplication for pairwise mixture screening.
"""

from typing import List, Dict, Any, Tuple
import torch
import numpy as np

from flavri.featurizer import mol2graph, BatchMolGraph
from flavri.constants import NUM_CHANNELS


class SinglesCollator:
    """Collates a list of single molecule samples."""
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        smiles_list = [item["smiles"] for item in batch]
        targets = torch.stack([item["target"] for item in batch], dim=0).to(self.device)
        masks = torch.stack([item["mask"] for item in batch], dim=0).to(self.device)
        batch_graph = mol2graph(smiles_list, device=self.device)

        return {
            "smiles": smiles_list,
            "graphs": batch_graph,
            "targets": targets,
            "masks": masks,
            "batch_size": len(batch),
        }


class PairsCollator:
    """
    Collates a list of pairwise mixture samples using O(1) Unique-SMILES Deduplication.
    Optionally samples synthetic identity self-pairs (A, A, delta=0) during training
    to enforce identity self-interaction neutrality.
    """
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        add_identity_pairs: bool = False,
        identity_ratio: float = 0.1,
    ):
        self.device = device
        self.add_identity_pairs = add_identity_pairs
        self.identity_ratio = identity_ratio

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        smiles_a = [item["smiles_a"] for item in batch]
        smiles_b = [item["smiles_b"] for item in batch]
        deltas = [item["delta"] for item in batch]
        masks = [item["mask"] for item in batch]
        cold_types = [item.get("cold_type", "train") for item in batch]

        # Extract unique SMILES across both constituent sets
        unique_smiles_map: Dict[str, int] = {}
        unique_smiles_list: List[str] = []

        for s in smiles_a + smiles_b:
            if s not in unique_smiles_map:
                unique_smiles_map[s] = len(unique_smiles_list)
                unique_smiles_list.append(s)

        idx_a = [unique_smiles_map[s] for s in smiles_a]
        idx_b = [unique_smiles_map[s] for s in smiles_b]

        # Synthetic identity pairs: (A, A, delta=0, mask=all 1)
        if self.add_identity_pairs and len(unique_smiles_list) > 0:
            num_id = max(1, int(len(batch) * self.identity_ratio))
            sampled_indices = np.random.choice(len(unique_smiles_list), size=num_id, replace=True)
            for s_idx in sampled_indices:
                idx_a.append(int(s_idx))
                idx_b.append(int(s_idx))
                deltas.append(torch.zeros(NUM_CHANNELS, dtype=torch.float32))
                # For synthetic identity, mask all channels so model learns zero across all dimensions
                masks.append(torch.ones(NUM_CHANNELS, dtype=torch.float32))
                cold_types.append("identity")

        # Convert to tensors
        idx_a_tensor = torch.tensor(idx_a, dtype=torch.long, device=self.device)
        idx_b_tensor = torch.tensor(idx_b, dtype=torch.long, device=self.device)
        deltas_tensor = torch.stack(deltas, dim=0).to(self.device)
        masks_tensor = torch.stack(masks, dim=0).to(self.device)

        # Batch encode unique molecules only once
        unique_graphs = mol2graph(unique_smiles_list, device=self.device)

        return {
            "unique_graphs": unique_graphs,
            "unique_smiles": unique_smiles_list,
            "idx_a": idx_a_tensor,
            "idx_b": idx_b_tensor,
            "deltas": deltas_tensor,
            "masks": masks_tensor,
            "cold_types": cold_types,
            "batch_size": len(idx_a),
        }
