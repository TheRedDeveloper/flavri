"""
flavri.losses
Unified Multimodal Chemosensory Loss Formulation:
1. Asymmetric Loss (ASL) for extreme multi-label sparsity on odor channels.
2. Top-K Margin Ranking Loss for precision and ranking order.
3. Smooth L1 / Huber regression for continuous taste deltas.
4. FlavriSinglesLoss for Stage 1.
5. FlavriMultimodalLoss & FlavriJointMultiTaskLoss for Stage 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flavri.constants import NUM_TASTE, NUM_ODOR, NUM_CHANNELS


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label sparse classification.
    - gamma_pos = 0.0: positive descriptors receive full unattenuated gradient.
    - gamma_neg = 2.0: easy negative descriptors are suppressed.
    - margin = 0.05: probability shift to eliminate easy background noise.
    """
    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 0.0,
        margin: float = 0.05,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin = margin
        self.eps = eps

    def forward(
        self,
        pred_probs: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        p = pred_probs.clamp(min=self.eps, max=1.0 - self.eps)
        targets = targets.clamp(min=0.0, max=1.0)

        # Positive branch
        pos_loss = -targets * torch.log(p)
        if self.gamma_pos > 0:
            pos_loss = pos_loss * ((1.0 - p) ** self.gamma_pos)

        # Negative branch with margin
        p_m = (p - self.margin).clamp(min=0.0, max=1.0 - self.eps)
        neg_loss = -(1.0 - targets) * (p_m ** self.gamma_neg) * torch.log(1.0 - p_m + self.eps)

        loss = (pos_loss + neg_loss) * masks
        active_counts = masks.sum(dim=-1).clamp(min=1.0)
        return (loss.sum(dim=-1) / active_counts).mean()


class TopKRankingLoss(nn.Module):
    """
    Enforces positive descriptors to have higher predicted probabilities than
    the hardest false-alarm negatives by at least `margin`.
    Directly maximizes Top-K precision and AUROC.
    """
    def __init__(self, margin: float = 0.15, top_k_neg: int = 5):
        super().__init__()
        self.margin = margin
        self.top_k_neg = top_k_neg

    def forward(
        self,
        pred_probs: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        B = pred_probs.shape[0]
        sample_losses = []
        for i in range(B):
            m_i = masks[i] > 0
            if m_i.sum() < 2:
                continue
            t_i = targets[i][m_i]
            p_i = pred_probs[i][m_i]

            pos_mask = t_i >= 0.5
            neg_mask = t_i < 0.5

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                p_pos = p_i[pos_mask]
                p_neg = p_i[neg_mask]
                k = min(self.top_k_neg, len(p_neg))
                top_neg, _ = torch.topk(p_neg, k)

                # Pairwise difference: p_pos.unsqueeze(1) - top_neg.unsqueeze(0)
                diff = p_pos.unsqueeze(1) - top_neg.unsqueeze(0)
                violations = F.relu(self.margin - diff)
                sample_losses.append(violations.mean())

        if sample_losses:
            return torch.stack(sample_losses).mean()
        return torch.tensor(0.0, device=pred_probs.device)


class FlavriSinglesLoss(nn.Module):
    """
    Upgraded loss for Stage 1 (single molecules):
    Combines Asymmetric Loss (ASL) + Top-K Ranking Loss.
    """
    def __init__(self, rank_weight: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.asl = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, margin=0.05, eps=eps)
        self.ranking = TopKRankingLoss(margin=0.15, top_k_neg=5)
        self.rank_weight = rank_weight

    def forward(
        self,
        pred_probs: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        l_asl = self.asl(pred_probs, targets, masks)
        l_rank = self.ranking(pred_probs, targets, masks)
        return l_asl + self.rank_weight * l_rank


class FlavriMultimodalLoss(nn.Module):
    """
    Loss for mixture interactions and blend profiles.
    - Huber on continuous taste deltas.
    - ASL / BCE on composite blend odor channels with negative attenuation.
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
        pred_probs: torch.Tensor,    # (B, 148) blend predictions in [0, 1]
        pred_deltas: torch.Tensor,   # (B, 148) synergy deltas in [-1, +1]
        targets: torch.Tensor,       # (B, 148) blend targets in [0, 1]
        masks: torch.Tensor,         # (B, 148) active masks
        is_mixture: torch.Tensor,    # (B,) bool
        deltas_true: torch.Tensor = None,  # (B, 148) delta targets
    ) -> torch.Tensor:
        B, C = pred_probs.shape
        device = pred_probs.device

        # 1. Classification BCE loss for odor blend profiles
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

        # 2. Continuous Huber loss for taste deltas
        delta_tgt = deltas_true if deltas_true is not None else targets
        huber_loss = F.smooth_l1_loss(pred_deltas, delta_tgt, beta=0.1, reduction="none")

        taste_mask = torch.zeros(C, dtype=torch.bool, device=device)
        taste_mask[:self.num_taste] = True
        mix_taste_mask = is_mix_expanded & taste_mask.unsqueeze(0)

        raw_loss = torch.where(mix_taste_mask, huber_loss, bce_loss)
        masked_loss = raw_loss * masks
        active_counts = masks.sum(dim=-1).clamp(min=1.0)
        return (masked_loss.sum(dim=-1) / active_counts).mean()
