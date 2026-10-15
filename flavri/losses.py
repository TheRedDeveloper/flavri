"""
flavri.losses
FLAVRI 2.0 Calibrated Multi-Objective Loss Formulation:
1. Class-Balanced Positive-Weighted Focal Loss (Global AUROC Calibration).
2. Plackett-Luce Listwise Ranking Loss (Top-5 Scent Precision).
3. Data-Driven Hierarchy & Antagonism Constraints (Ontology Regularization).
4. Persistent Chemical Reconstruction Loss (Pretext Anchor).
5. Continuous Huber Loss for Taste Synergy Deltas.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from flavri.constants import NUM_TASTE, NUM_ODOR, NUM_CHANNELS

REPO_ROOT = Path(__file__).resolve().parent.parent
H_PATH = REPO_ROOT / "data" / "quarantining" / "hierarchy_matrix.pt"
A_PATH = REPO_ROOT / "data" / "quarantining" / "antagonism_matrix.pt"


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Positive-Weighted Focal BCE (No Margin Floor).
    - True negatives are driven cleanly toward zero (restores global AUROC).
    - Sparse positives receive frequency-based gradient boost (w_pos).
    - Focal gamma = 1.0 focuses learning on hard examples without truncating negatives.
    """
    def __init__(self, gamma: float = 1.0, max_pos_weight: float = 8.0, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.max_pos_weight = max_pos_weight
        self.eps = eps

    def forward(
        self,
        pred_probs: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        pos_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p = pred_probs.clamp(min=self.eps, max=1.0 - self.eps)
        targets = targets.clamp(min=0.0, max=1.0)

        # Positive branch with focal factor
        pos_focal = (1.0 - p) ** self.gamma
        pos_loss = -targets * pos_focal * torch.log(p)
        if pos_weights is not None:
            pos_loss = pos_loss * pos_weights.unsqueeze(0)

        # Negative branch with focal factor
        neg_focal = p ** self.gamma
        neg_loss = -(1.0 - targets) * neg_focal * torch.log(1.0 - p)

        loss = (pos_loss + neg_loss) * masks
        active_counts = masks.sum(dim=-1).clamp(min=1.0)
        return (loss.sum(dim=-1) / active_counts).mean()


class PlackettLuceRankingLoss(nn.Module):
    """
    Plackett-Luce Listwise Ranking Loss:
    Treats multi-label odor descriptors as a probability distribution over scent dominance.
    Minimizes KL divergence between true normalized distribution and predicted softmax distribution.
    Drives Top-5 Precision in O(C) vectorized GPU operations.
    """
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        odor_probs: torch.Tensor,
        odor_targets: torch.Tensor,
        odor_masks: torch.Tensor,
    ) -> torch.Tensor:
        # Pseudo-logits from probabilities: s = log(p / (1 - p))
        p_c = odor_probs.clamp(min=1e-5, max=1.0 - 1e-5)
        pseudo_logits = torch.log(p_c / (1.0 - p_c))

        # Predicted log-softmax distribution over odor channels
        log_pred = F.log_softmax(pseudo_logits / self.tau, dim=-1)

        # True target distribution (normalized active positives)
        active_targets = (odor_targets * odor_masks).clamp(min=0.0)
        sum_targets = active_targets.sum(dim=-1, keepdim=True)
        has_positives = (sum_targets.squeeze(-1) > 0)

        p_true = active_targets / sum_targets.clamp(min=1.0)

        # KL divergence: sum(p_true * (log(p_true) - log_pred))
        # When p_true > 0: -p_true * log_pred
        kl = - (p_true * log_pred).sum(dim=-1)

        if has_positives.sum() > 0:
            return kl[has_positives].mean()
        return torch.tensor(0.0, device=odor_probs.device)


class OntologyConstraintLoss(nn.Module):
    """
    Enforces Hierarchy consistency and Antagonism mutual exclusion on odor channels.
    H[parent, child] = 1.0 -> Penalizes if p_child > p_parent
    A[i, j] = 1.0          -> Penalizes joint activation p_i * p_j
    """
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.has_matrices = False

        if H_PATH.exists() and A_PATH.exists():
            H = torch.load(str(H_PATH), map_location=device, weights_only=False)
            A = torch.load(str(A_PATH), map_location=device, weights_only=False)
            p_idx, c_idx = torch.where(H > 0)
            self.register_buffer("parent_idx", p_idx)
            self.register_buffer("child_idx", c_idx)
            self.register_buffer("antagonism_matrix", A)
            self.has_matrices = True

    def forward(self, odor_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.has_matrices:
            zero = torch.tensor(0.0, device=odor_probs.device)
            return zero, zero

        # 1. Hierarchy consistency: ReLU(p_child - p_parent)^2
        p_parents = odor_probs[:, self.parent_idx]
        p_children = odor_probs[:, self.child_idx]
        hier_violations = F.relu(p_children - p_parents)
        loss_hier = (hier_violations ** 2).mean()

        # 2. Antagonism mutual exclusion: 0.5 * mean( (p * A) . p )
        antag_joint = torch.matmul(odor_probs, self.antagonism_matrix) * odor_probs
        loss_antag = 0.5 * antag_joint.sum(dim=-1).mean()

        return loss_hier, loss_antag


class ChemicalReconstructionLoss(nn.Module):
    """
    Heteroatom-Calibrated Chemical Reconstruction Loss.
    Replaces unweighted MSE across 133 sparse dims with class-weighted multi-task cross-entropy.
    Heavily penalizes misidentifying key aroma/taste heteroatoms (N, O, S, P, halogens)
    and predicts physical substructure (degree, formal charge, hydrogens, hybridization, aromaticity).
    """
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        weights = torch.full((101,), 0.5, device=device)
        weights[5] = 1.0   # Carbon (baseline)
        weights[6] = 6.0   # Nitrogen (amines, pyrazines)
        weights[7] = 6.0   # Oxygen (esters, aldehydes, ketones)
        weights[8] = 6.0   # Fluorine
        weights[14] = 8.0  # Phosphorus
        weights[15] = 10.0 # Sulfur (thiols, roasted notes)
        weights[16] = 6.0  # Chlorine
        weights[34] = 6.0  # Bromine
        weights[52] = 6.0  # Iodine
        self.register_buffer("elem_weights", weights)

    def forward(
        self,
        pred_atom_features: torch.Tensor,
        true_atom_features: torch.Tensor,
        masked_atom_indices: Optional[torch.Tensor] = None,
        return_accuracies: bool = False,
    ):
        if masked_atom_indices is not None and masked_atom_indices.sum() > 0:
            preds = pred_atom_features[masked_atom_indices]
            targets = true_atom_features[masked_atom_indices]
        else:
            preds = pred_atom_features
            targets = true_atom_features

        # 1. Element Cross-Entropy (dims 0..101) with heteroatom reweighting
        target_elem = targets[:, :101].argmax(dim=-1)
        loss_elem = F.cross_entropy(preds[:, :101], target_elem, weight=self.elem_weights)

        # 2. Structural Attribute Cross-Entropies
        loss_degree = F.cross_entropy(preds[:, 101:108], targets[:, 101:108].argmax(dim=-1))
        loss_charge = F.cross_entropy(preds[:, 108:114], targets[:, 108:114].argmax(dim=-1))
        loss_hs = F.cross_entropy(preds[:, 119:125], targets[:, 119:125].argmax(dim=-1))
        loss_hybrid = F.cross_entropy(preds[:, 125:131], targets[:, 125:131].argmax(dim=-1))
        loss_arom = F.binary_cross_entropy_with_logits(preds[:, 131], targets[:, 131])

        total_loss = loss_elem + 0.3 * (loss_degree + loss_charge + loss_hs + loss_hybrid + loss_arom)

        if return_accuracies:
            with torch.no_grad():
                pred_elem = preds[:, :101].argmax(dim=-1)
                elem_acc = (pred_elem == target_elem).float().mean().item()
                is_hetero = (target_elem != 5)  # Not Carbon
                if is_hetero.sum() > 0:
                    hetero_acc = (pred_elem[is_hetero] == target_elem[is_hetero]).float().mean().item()
                else:
                    hetero_acc = 1.0
            return total_loss, elem_acc, hetero_acc

        return total_loss


class FlavriSinglesLoss(nn.Module):
    """
    FLAVRI 2.0 Unified Single Molecule Loss:
    L = L_calib + lambda_rank * L_rank + lambda_hier * L_hier + lambda_antag * L_antag
    """
    def __init__(
        self,
        rank_weight: float = 0.5,
        hier_weight: float = 0.1,
        antag_weight: float = 0.05,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.calib_loss = ClassBalancedFocalLoss(gamma=1.0)
        self.rank_loss = PlackettLuceRankingLoss(tau=1.0)
        self.ontology_loss = OntologyConstraintLoss(device=device)
        self.rank_weight = rank_weight
        self.hier_weight = hier_weight
        self.antag_weight = antag_weight

    def forward(
        self,
        pred_probs: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Global calibrated focal loss across all 148 channels
        l_calib = self.calib_loss(pred_probs, targets, masks)

        # Odor-specific channels (11..147)
        odor_p = pred_probs[:, NUM_TASTE:]
        odor_t = targets[:, NUM_TASTE:]
        odor_m = masks[:, NUM_TASTE:]

        # 2. Plackett-Luce Listwise Ranking loss
        l_rank = self.rank_loss(odor_p, odor_t, odor_m)

        # 3. Ontology Hierarchy & Antagonism constraints
        l_hier, l_antag = self.ontology_loss(odor_p)

        total = l_calib + self.rank_weight * l_rank + self.hier_weight * l_hier + self.antag_weight * l_antag
        return total


class FlavriMultimodalLoss(nn.Module):
    """
    FLAVRI 2.0 Mixture Synergy & Blend Loss:
    - Continuous Huber regression directly on taste & odor synergy deltas to ensure
      linear correlation (Pearson r) and magnitude ranking.
    - Calibrated BCE on composite blend profiles with negative attenuation.
    """
    def __init__(
        self,
        mix_neg_weight: float = 0.2,
        odor_delta_weight: float = 0.5,
        eps: float = 1e-7,
        num_taste: int = NUM_TASTE,
        num_odor: int = NUM_ODOR,
    ):
        super().__init__()
        self.mix_neg_weight = mix_neg_weight
        self.odor_delta_weight = odor_delta_weight
        self.eps = eps
        self.num_taste = num_taste
        self.num_odor = num_odor

    def forward(
        self,
        pred_probs: torch.Tensor,    # (B, 148) blend predictions in [0, 1]
        pred_deltas: torch.Tensor,   # (B, 148) synergy deltas in [-1, +1]
        targets: torch.Tensor,       # (B, 148) blend targets in [0, 1]
        masks: torch.Tensor,         # (B, 148) active masks
        is_mixture: torch.Tensor,    # (B,) bool
        deltas_true: torch.Tensor = None,  # (B, 148) delta targets
    ) -> torch.Tensor:
        B, C = pred_probs.shape
        device = pred_probs.device

        # 1. Classification BCE loss for blend profiles
        t_bce = targets.clamp(min=0.0, max=1.0)
        p = pred_probs.clamp(min=self.eps, max=1.0 - self.eps)
        pos_loss = -t_bce * torch.log(p)
        neg_loss = -(1.0 - t_bce) * torch.log(1.0 - p)

        neg_weight = torch.ones((B, C), device=device)
        is_mix_expanded = is_mixture.unsqueeze(1).expand(B, C)

        odor_mask = torch.zeros(C, dtype=torch.bool, device=device)
        odor_mask[self.num_taste:] = True
        mix_odor_mask = is_mix_expanded & odor_mask.unsqueeze(0)
        neg_weight = torch.where(mix_odor_mask, torch.tensor(self.mix_neg_weight, device=device), neg_weight)

        bce_loss = pos_loss + neg_weight * neg_loss

        # 2. Continuous Huber loss for synergy deltas
        delta_tgt = deltas_true if deltas_true is not None else targets
        huber_loss = F.smooth_l1_loss(pred_deltas, delta_tgt, beta=0.1, reduction="none")

        taste_mask = torch.zeros(C, dtype=torch.bool, device=device)
        taste_mask[:self.num_taste] = True
        mix_taste_mask = is_mix_expanded & taste_mask.unsqueeze(0)

        # For taste mixture channels: Huber loss on delta
        # For odor mixture channels: BCE blend classification + odor_delta_weight * Huber delta regression
        odor_combined = bce_loss + self.odor_delta_weight * huber_loss
        raw_loss = torch.where(mix_taste_mask, huber_loss, odor_combined)

        masked_loss = raw_loss * masks
        active_counts = masks.sum(dim=-1).clamp(min=1.0)
        return (masked_loss.sum(dim=-1) / active_counts).mean()
