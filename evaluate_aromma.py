"""
evaluate_aromma.py
Evaluates AROMMA (aromma_best_fold.pt) on FLAVRI's zero-leakage holdout test sets:
1. singles_test.csv (1,590 molecules)
2. pair_test.csv (10,492 mixtures: half_cold & double_cold)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, roc_auc_score

AROMMA_DIR = Path("c:/Users/RedDev/clones/aromma-mcp")
FLAVRI_DIR = Path("c:/Users/RedDev/clones/flavri")

sys.path.insert(0, str(AROMMA_DIR / "src"))
sys.path.insert(0, str(FLAVRI_DIR))

from aromma_mcp.model import get_predictor
from aromma_mcp.vocab import get_vocabulary
from flavri.constants import CANONICAL_ODOR_DESCRIPTORS, ODOR_CHANNELS, ODOR_DESCRIPTOR_MAP
from data.collection.odor_ingestion.constants import SYNONYM_MAP

def main():
    print("=" * 60)
    print("AROMMA ZERO-LEAKAGE BENCHMARK EVALUATION")
    print("=" * 60)

    # 1. Initialize AROMMA Predictor
    print("Loading AROMMA predictor and vocabulary...")
    predictor = get_predictor()
    model = predictor.model
    device = predictor.device
    vocab = get_vocabulary()
    label_to_idx = {lbl: i for i, lbl in enumerate(vocab.labels)}
    print(f"Loaded AROMMA on device: {device} ({len(vocab.labels)} descriptors)")

    # Build mapping from FLAVRI odor channels to AROMMA indices
    syn_rev = {}
    for syn, target in SYNONYM_MAP.items():
        syn_rev.setdefault(target, []).append(syn)

    odor_ch_to_aromma = {}
    for d in CANONICAL_ODOR_DESCRIPTORS:
        ch = ODOR_DESCRIPTOR_MAP[d]
        main_idx = label_to_idx[d]
        syn_idxs = [label_to_idx[s] for s in syn_rev.get(d, [])]
        odor_ch_to_aromma[ch] = (main_idx, syn_idxs)

    # ----------------------------------------------------
    # 2. Evaluate Singles Test Set (1,590 molecules)
    # ----------------------------------------------------
    singles_csv = FLAVRI_DIR / "data/quarantining/singles_test.csv"
    print(f"\n--- Evaluating Singles Test ({singles_csv.name}) ---")
    df_singles = pd.read_csv(singles_csv)
    smiles_list = df_singles["smiles"].tolist()

    batch_size = 128
    all_p_aromma = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            chunk = smiles_list[i : i + batch_size]
            logits = model.forward(chunk)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_p_aromma.append(probs)

    p_aromma_152 = np.concatenate(all_p_aromma, axis=0)

    # Map to FLAVRI 137 odor channels
    pred_odor_137 = np.zeros((len(smiles_list), len(ODOR_CHANNELS)), dtype=np.float32)
    for c_idx, ch in enumerate(ODOR_CHANNELS):
        main_idx, syn_idxs = odor_ch_to_aromma[ch]
        vals = p_aromma_152[:, main_idx]
        for s_idx in syn_idxs:
            vals = np.maximum(vals, p_aromma_152[:, s_idx])
        pred_odor_137[:, c_idx] = vals

    target_cols = [f"y_{ch}" for ch in ODOR_CHANNELS]
    mask_cols = [f"mask_{ch}" for ch in ODOR_CHANNELS]
    y_true_odor = df_singles[target_cols].to_numpy()
    mask_odor = df_singles[mask_cols].to_numpy()

    # Macro & Micro AUROC
    aurocs = []
    for c in range(len(ODOR_CHANNELS)):
        m_c = mask_odor[:, c] > 0
        if m_c.sum() >= 10 and len(np.unique(y_true_odor[m_c, c])) > 1:
            try:
                aurocs.append(roc_auc_score(y_true_odor[m_c, c], pred_odor_137[m_c, c]))
            except Exception:
                pass

    aromma_odor_macro_auroc = float(np.mean(aurocs)) if aurocs else 0.0

    active_mask = mask_odor > 0
    if len(np.unique(y_true_odor[active_mask])) > 1:
        aromma_odor_micro_auroc = float(roc_auc_score(y_true_odor[active_mask], pred_odor_137[active_mask]))
    else:
        aromma_odor_micro_auroc = 0.0

    # Top-5 Precision
    top5_precs = []
    for i in range(len(smiles_list)):
        if mask_odor[i].sum() > 0:
            top5_idx = np.argsort(pred_odor_137[i])[-5:]
            top5_precs.append(y_true_odor[i, top5_idx].sum() / 5.0)
    aromma_top5_prec = float(np.mean(top5_precs)) if top5_precs else 0.0

    print(f"AROMMA Singles Odor Macro AUROC: {aromma_odor_macro_auroc:.4f}")
    print(f"AROMMA Singles Odor Micro AUROC: {aromma_odor_micro_auroc:.4f}")
    print(f"AROMMA Singles Odor Top-5 Prec:  {aromma_top5_prec:.4f}")

    # ----------------------------------------------------
    # 3. Evaluate Pairs Test Set (10,492 pairs)
    # ----------------------------------------------------
    pairs_csv = FLAVRI_DIR / "data/quarantining/pair_test.csv"
    print(f"\n--- Evaluating Pairs Test ({pairs_csv.name}) ---")
    df_pairs = pd.read_csv(pairs_csv)
    pairs_a = df_pairs["smiles_a"].tolist()
    pairs_b = df_pairs["smiles_b"].tolist()
    cold_types = df_pairs["cold_type"].tolist()

    # Precompute constituent single predictions
    unique_pair_smiles = list(set(pairs_a + pairs_b))
    print(f"Encoding {len(unique_pair_smiles)} unique constituent molecules...")
    smi_to_p = {}
    with torch.no_grad():
        for i in range(0, len(unique_pair_smiles), batch_size):
            chunk = unique_pair_smiles[i : i + batch_size]
            logits = model.forward(chunk)
            probs = torch.sigmoid(logits).cpu().numpy()
            for smi, p in zip(chunk, probs):
                smi_to_p[smi] = p

    # Predict mixtures
    print(f"Computing {len(pairs_a)} mixture predictions...")
    mix_strings = [f"{a};{b}" for a, b in zip(pairs_a, pairs_b)]
    all_p_mix = []
    with torch.no_grad():
        for i in range(0, len(mix_strings), batch_size):
            chunk = mix_strings[i : i + batch_size]
            logits = model.forward(chunk)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_p_mix.append(probs)

    p_mix_152 = np.concatenate(all_p_mix, axis=0)

    # Compute deltas: Delta = P_mix - 0.5*(P_a + P_b)
    deltas_152 = np.zeros_like(p_mix_152)
    for i, (a, b) in enumerate(zip(pairs_a, pairs_b)):
        deltas_152[i] = p_mix_152[i] - 0.5 * (smi_to_p[a] + smi_to_p[b])

    # Map deltas to FLAVRI 137 Odor Channels
    pred_delta_137 = np.zeros((len(df_pairs), len(ODOR_CHANNELS)), dtype=np.float32)
    for c_idx, ch in enumerate(ODOR_CHANNELS):
        main_idx, _ = odor_ch_to_aromma[ch]
        pred_delta_137[:, c_idx] = deltas_152[:, main_idx]

    pair_target_cols = [f"delta_{ch}" for ch in ODOR_CHANNELS]
    pair_mask_cols = [f"mask_{ch}" for ch in ODOR_CHANNELS]
    t_delta = df_pairs[pair_target_cols].to_numpy()
    p_mask = df_pairs[pair_mask_cols].to_numpy()

    for cohort in ["half_cold", "double_cold"]:
        idx_cohort = [i for i, ct in enumerate(cold_types) if ct == cohort]
        p_act = pred_delta_137[idx_cohort][p_mask[idx_cohort] > 0]
        t_act = t_delta[idx_cohort][p_mask[idx_cohort] > 0]

        r_val, _ = pearsonr(p_act, t_act) if len(p_act) > 1 else (0.0, 0.0)
        rho_val, _ = spearmanr(p_act, t_act) if len(p_act) > 1 else (0.0, 0.0)
        f1_val = f1_score((np.abs(t_act) >= 0.1).astype(int), (np.abs(p_act) >= 0.1).astype(int), zero_division=0)
        
        # Direction accuracy on non-zero deltas
        sig = np.abs(t_act) >= 0.05
        dir_acc = (np.sign(p_act[sig]) == np.sign(t_act[sig])).mean() if sig.sum() > 0 else 1.0

        # Blend MAE
        # True blend = clip(0.5*(P_a + P_b) + t_delta, 0, 1)
        # Pred blend = p_mix
        idx_arr = np.array(idx_cohort)
        pred_mix_137 = np.zeros((len(idx_cohort), len(ODOR_CHANNELS)), dtype=np.float32)
        true_linear_137 = np.zeros((len(idx_cohort), len(ODOR_CHANNELS)), dtype=np.float32)
        for k, orig_idx in enumerate(idx_cohort):
            a_smi = pairs_a[orig_idx]
            b_smi = pairs_b[orig_idx]
            for c_idx, ch in enumerate(ODOR_CHANNELS):
                main_idx, _ = odor_ch_to_aromma[ch]
                pred_mix_137[k, c_idx] = p_mix_152[orig_idx, main_idx]
                true_linear_137[k, c_idx] = 0.5 * (smi_to_p[a_smi][main_idx] + smi_to_p[b_smi][main_idx])
        
        true_blend_137 = np.clip(true_linear_137 + t_delta[idx_arr], 0.0, 1.0)
        mask_cohort = p_mask[idx_arr] > 0
        blend_mae = np.mean(np.abs(pred_mix_137[mask_cohort] - true_blend_137[mask_cohort]))

        print(f"\nAROMMA [{cohort} ({len(idx_cohort)} pairs)]:")
        print(f"  Blend MAE:        {blend_mae:.4f}")
        print(f"  Pearson r:        {r_val:.4f}")
        print(f"  Spearman rho:     {rho_val:.4f}")
        print(f"  Synergy F1:       {f1_val:.4f}")
        print(f"  Direction Acc:    {dir_acc:.2%}")

if __name__ == "__main__":
    main()
