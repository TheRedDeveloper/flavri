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
    FLAVRI 2.0 Pure Interaction Synergy Loss:
    Direct Continuous Huber Regression on taste and odor synergy deltas across all channels:
      L = (1 / |M|) sum_{c in M} SmoothL1(pred_deltas_c, true_deltas_c, beta=0.1)
    Treats taste and odor symmetrically and without artificial blend BCE artifacts.
    """
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pred_deltas: torch.Tensor,
        deltas_true: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        huber_loss = F.smooth_l1_loss(pred_deltas, deltas_true, beta=self.beta, reduction="none")
        masked_loss = huber_loss * masks
        active_counts = masks.sum(dim=-1).clamp(min=1.0)
        return (masked_loss.sum(dim=-1) / active_counts).mean()
