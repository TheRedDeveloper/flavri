"""
flavri.losses
Unified Multimodal Chemosensory Loss Formulation (FlavriMultimodalLoss).
Implements universal dense odor masking with negative attenuation (w_neg=0.2),
Huber loss for taste synergy deltas, and identity regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flavri.constants import NUM_TASTE, NUM_ODOR, NUM_CHANNELS


class FlavriMultimodalLoss(nn.Module):
    """
    Unified Multimodal Chemosensory Loss Function (148 Channels)
    
    Handles:
    1. Masked Binary Cross-Entropy across active channels.
    2. Differentiated negative attenuation (w_neg=0.2) for sparse odor blend tags.
    3. Huber / Smooth-L1 penalty (delta=0.1) for continuous taste synergy deltas.
    4. Exact zero penalty for unmeasured/masked channels (mask=0).
    5. Per-sample normalization by active channel count.
    """
    def __init__(
        self,
        mix_neg_weight: float = 0.2,
        eps: float = 1e-7,
        num_taste: int = NUM_TASTE,
        num_odor: int = NUM_ODOR,
    ):
        super().__init__()
        self.mix_neg_weight = mix_neg_weight
        self.eps = eps
        self.num_taste = num_taste
        self.num_odor = num_odor
        self.total_channels = num_taste + num_odor

    def forward(
        self,
        pred_probs: torch.Tensor,    # (B, 148) in [0, 1]
        pred_deltas: torch.Tensor,   # (B, 148) in [-1, +1] (from synergy head)
        targets: torch.Tensor,       # (B, 148) targets (probabilities or deltas)
        masks: torch.Tensor,         # (B, 148) in {0, 1}
        is_mixture: torch.Tensor,    # (B,) bool
    ) -> torch.Tensor:
        B, C = pred_probs.shape
        device = pred_probs.device

        # 1. Classification BCE loss for singles and odor blends
        p = pred_probs.clamp(min=self.eps, max=1.0 - self.eps)
        pos_loss = -targets * torch.log(p)
        neg_loss = -(1.0 - targets) * torch.log(1.0 - p)

        # Weighting: 1.0 for singles; mix_neg_weight (0.2) for odor mixtures
        neg_weight = torch.ones((B, C), device=device)
        is_mix_expanded = is_mixture.unsqueeze(1).expand(B, C)

        # Apply negative attenuation only to odor channels (11..147) in mixtures
        odor_mask = torch.zeros(C, dtype=torch.bool, device=device)
        odor_mask[self.num_taste:] = True
        mix_odor_mask = is_mix_expanded & odor_mask.unsqueeze(0)
        neg_weight = torch.where(mix_odor_mask, torch.tensor(self.mix_neg_weight, device=device), neg_weight)

        bce_loss = pos_loss + neg_weight * neg_loss

        # 2. Continuous Huber loss for taste pairs (where targets are continuous deltas)
        huber_loss = F.smooth_l1_loss(pred_deltas, targets, beta=0.1, reduction="none")

        # Select loss branch: taste channels in mixtures use Huber; everything else uses BCE
        taste_mask = torch.zeros(C, dtype=torch.bool, device=device)
        taste_mask[:self.num_taste] = True
        mix_taste_mask = is_mix_expanded & taste_mask.unsqueeze(0)

        raw_loss = torch.where(mix_taste_mask, huber_loss, bce_loss)

        # 3. Apply channel mask
        masked_loss = raw_loss * masks

        # 4. Normalize by active channel count per sample
        active_counts = masks.sum(dim=-1).clamp(min=1.0)
        sample_loss = masked_loss.sum(dim=-1) / active_counts

        return sample_loss.mean()


class FlavriSinglesLoss(nn.Module):
    """
    Dedicated loss for Stage 1 (single molecules only).
    Masked Binary Cross-Entropy over active channels.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_probs: torch.Tensor,  # (B, 148)
        targets: torch.Tensor,     # (B, 148)
        masks: torch.Tensor,       # (B, 148)
    ) -> torch.Tensor:
        p = pred_probs.clamp(min=self.eps, max=1.0 - self.eps)
        bce = -targets * torch.log(p) - (1.0 - targets) * torch.log(1.0 - p)
        masked_bce = bce * masks
        active_counts = masks.sum(dim=-1).clamp(min=1.0)
        return (masked_bce.sum(dim=-1) / active_counts).mean()
