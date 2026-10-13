"""
flavri.evaluate
Evaluation benchmark suite across Cold-Start Holdout cohorts:
1. Singles Test (Odor): Macro & Micro AUROC, Top-5 Precision
2. Singles Test (Taste): AUROC across 11 taste channels
3. Pairs Test (half_cold): Pearson r, Spearman rho, Synergy F1
4. Pairs Test (double_cold): Blend MAE, Delta Direction Accuracy
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr, spearmanr

from flavri.constants import (
    NUM_CHANNELS,
    NUM_TASTE,
    NUM_ODOR,
    ALL_CHANNELS,
    TASTE_CHANNELS,
    ODOR_CHANNELS,
)
from flavri.data import FlavriSinglesDataset, FlavriPairsDataset, SinglesCollator, PairsCollator
from flavri.models import FlavriMultimodalModel


@torch.no_grad()
def evaluate_singles(
    model: FlavriMultimodalModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates model on singles dataset (e.g. singles_test.csv).
    Computes:
      - Odor Macro AUROC & Micro AUROC
      - Odor Top-5 Precision
      - Taste Macro AUROC
      - Global Masked BCE Loss
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    for batch in dataloader:
        graphs = batch["graphs"]
        targets = batch["targets"].cpu().numpy()
        masks = batch["masks"].cpu().numpy()

        _, preds = model.predict_singles(graphs)
        preds = preds.detach().cpu().numpy()

        all_preds.append(preds)
        all_targets.append(targets)
        all_masks.append(masks)

    preds_arr = np.concatenate(all_preds, axis=0)      # (N, 148)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 148)
    masks_arr = np.concatenate(all_masks, axis=0)      # (N, 148)

    # 1. Taste metrics (channels 0..10)
    taste_preds = preds_arr[:, :NUM_TASTE]
    taste_targets = targets_arr[:, :NUM_TASTE]
    taste_masks = masks_arr[:, :NUM_TASTE]

    taste_aurocs = []
    for c in range(NUM_TASTE):
        mask_c = taste_masks[:, c] > 0
        if mask_c.sum() >= 10:
            y_true = taste_targets[mask_c, c]
            y_pred = taste_preds[mask_c, c]
            if len(np.unique(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    taste_aurocs.append(auc)
                except Exception:
                    pass

    taste_macro_auroc = float(np.mean(taste_aurocs)) if taste_aurocs else 0.0

    # 2. Odor metrics (channels 11..147)
    odor_preds = preds_arr[:, NUM_TASTE:]
    odor_targets = targets_arr[:, NUM_TASTE:]
    odor_masks = masks_arr[:, NUM_TASTE:]

    odor_aurocs = []
    for c in range(NUM_ODOR):
        mask_c = odor_masks[:, c] > 0
        if mask_c.sum() >= 10:
            y_true = odor_targets[mask_c, c]
            y_pred = odor_preds[mask_c, c]
            if len(np.unique(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    odor_aurocs.append(auc)
                except Exception:
                    pass

    odor_macro_auroc = float(np.mean(odor_aurocs)) if odor_aurocs else 0.0

    # Odor micro AUROC across all active entries
    odor_active = odor_masks > 0
    if odor_active.sum() > 0 and len(np.unique(odor_targets[odor_active])) > 1:
        try:
            odor_micro_auroc = float(roc_auc_score(odor_targets[odor_active], odor_preds[odor_active]))
        except Exception:
            odor_micro_auroc = 0.0
    else:
        odor_micro_auroc = 0.0

    # Odor Top-5 Precision
    top5_precisions = []
    for i in range(len(odor_preds)):
        if odor_masks[i].sum() > 0:
            top5_idx = np.argsort(odor_preds[i])[-5:]
            true_pos = odor_targets[i, top5_idx].sum()
            top5_precisions.append(true_pos / 5.0)

    odor_top5_prec = float(np.mean(top5_precisions)) if top5_precisions else 0.0

    return {
        "taste_macro_auroc": taste_macro_auroc,
        "odor_macro_auroc": odor_macro_auroc,
        "odor_micro_auroc": odor_micro_auroc,
        "odor_top5_prec": odor_top5_prec,
    }


@torch.no_grad()
def evaluate_pairs(
    model: FlavriMultimodalModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates model on mixture pairs dataset (pair_test.csv).
    Splits results into 'half_cold', 'double_cold', and 'overall'.
    Computes:
      - Pearson r
      - Spearman rho
      - Synergy F1 (binary classification: abs(delta) >= 0.1)
      - Blend MAE
      - Direction Accuracy (sign agreement on non-zero deltas)
    """
    model.eval()
    
    records: Dict[str, Dict[str, List]] = {
        "half_cold": {"pred_deltas": [], "true_deltas": [], "masks": [], "blend_preds": [], "blend_trues": []},
        "double_cold": {"pred_deltas": [], "true_deltas": [], "masks": [], "blend_preds": [], "blend_trues": []},
    }

    for batch in dataloader:
        unique_graphs = batch["unique_graphs"]
        idx_a = batch["idx_a"]
        idx_b = batch["idx_b"]
        true_deltas = batch["deltas"].cpu().numpy()
        masks = batch["masks"].cpu().numpy()
        cold_types = batch["cold_types"]

        pred_deltas, y_blend, v_a, v_b = model.predict_pairs_batch(unique_graphs, idx_a, idx_b)
        pred_deltas = pred_deltas.detach().cpu().numpy()
        y_blend = y_blend.detach().cpu().numpy()

        # Approximate blend target as 0.5 * V_A + 0.5 * V_B + true_delta clamped to [0, 1]
        v_a_np = v_a.detach().cpu().numpy()
        v_b_np = v_b.detach().cpu().numpy()
        true_blend = np.clip(0.5 * v_a_np + 0.5 * v_b_np + true_deltas, 0.0, 1.0)

        for i, ct in enumerate(cold_types):
            key = ct if ct in records else "half_cold"
            records[key]["pred_deltas"].append(pred_deltas[i])
            records[key]["true_deltas"].append(true_deltas[i])
            records[key]["masks"].append(masks[i])
            records[key]["blend_preds"].append(y_blend[i])
            records[key]["blend_trues"].append(true_blend[i])

    results = {}
    for cohort in ["half_cold", "double_cold"]:
        if not records[cohort]["pred_deltas"]:
            results[cohort] = {
                "pearson_r": 0.0,
                "spearman_rho": 0.0,
                "synergy_f1": 0.0,
                "blend_mae": 0.0,
                "direction_acc": 0.0,
            }
            continue

        p_delta = np.array(records[cohort]["pred_deltas"])
        t_delta = np.array(records[cohort]["true_deltas"])
        m = np.array(records[cohort]["masks"])
        p_blend = np.array(records[cohort]["blend_preds"])
        t_blend = np.array(records[cohort]["blend_trues"])

        # Active mask filter
        active_indices = m > 0
        p_act = p_delta[active_indices]
        t_act = t_delta[active_indices]

        # Pearson r & Spearman rho
        if len(p_act) > 1 and np.std(p_act) > 1e-6 and np.std(t_act) > 1e-6:
            r_val, _ = pearsonr(p_act, t_act)
            rho_val, _ = spearmanr(p_act, t_act)
        else:
            r_val, rho_val = 0.0, 0.0

        # Synergy F1 (active interaction defined as |delta| >= 0.1)
        bin_true = (np.abs(t_act) >= 0.1).astype(int)
        bin_pred = (np.abs(p_act) >= 0.1).astype(int)
        f1_val = float(f1_score(bin_true, bin_pred, zero_division=0))

        # Blend MAE
        blend_act_p = p_blend[active_indices]
        blend_act_t = t_blend[active_indices]
        blend_mae = float(np.mean(np.abs(blend_act_p - blend_act_t))) if len(blend_act_p) > 0 else 0.0

        # Direction Accuracy on non-zero true deltas (|t_delta| >= 0.05)
        sig_indices = np.abs(t_act) >= 0.05
        if sig_indices.sum() > 0:
            dir_correct = (np.sign(p_act[sig_indices]) == np.sign(t_act[sig_indices])).mean()
            dir_acc = float(dir_correct)
        else:
            dir_acc = 1.0

        results[cohort] = {
            "pearson_r": float(r_val),
            "spearman_rho": float(rho_val),
            "synergy_f1": f1_val,
            "blend_mae": blend_mae,
            "direction_acc": dir_acc,
        }

    return results
