"""
evaluate_flavri_complete.py
Comprehensive FLAVRI evaluation:
1. Quantitative test metrics on singles_test.csv and pair_test.csv
2. Detailed qualitative analysis:
   - CCN (Ethylamine) taste and odor profile
   - Reference singles (Vanillin, Caffeine, Menthol, Capsaicin, MSG, Limonene, etc.)
   - Coffee + Chocolate mixture pairings (constituent profiles, synergy deltas, composite blend)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr, spearmanr

FLAVRI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FLAVRI_DIR))

import argparse
from flavri.constants import (
    NUM_CHANNELS,
    NUM_TASTE,
    NUM_ODOR,
    ALL_CHANNELS,
    TASTE_CHANNELS,
    ODOR_CHANNELS,
    CANONICAL_ODOR_DESCRIPTORS,
)
from flavri.data import FlavriSinglesDataset, FlavriPairsDataset, SinglesCollator, PairsCollator
from flavri.featurizer import mol2graph
from flavri.models import FlavriMultimodalModel


def main():
    parser = argparse.ArgumentParser(description="Evaluate FLAVRI Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/stage3/swa_model.pt",
        help="Path to checkpoint to evaluate (e.g. checkpoints/stage1/best_model.pt, checkpoints/stage2/best_model.pt, or checkpoints/stage3/swa_model.pt)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden dimension (auto-detected from checkpoint if omitted)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("FLAVRI COMPREHENSIVE BENCHMARK & INFERENCE")
    print(f"Device: {device}")
    print("=" * 70)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = FLAVRI_DIR / ckpt_path
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    if args.hidden_dim is not None:
        hidden_dim = args.hidden_dim
    elif "hidden_dim" in checkpoint:
        hidden_dim = checkpoint["hidden_dim"]
    elif "backbone.W_m.weight" in state_dict:
        hidden_dim = state_dict["backbone.W_m.weight"].shape[0]
    else:
        hidden_dim = 512

    print(f"Model hidden_dim: {hidden_dim}")
    model = FlavriMultimodalModel(hidden_dim=hidden_dim, depth=6).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # =========================================================================
    # PART 1: Quantitative Holdout Test Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: QUANTITATIVE BENCHMARK ON ZERO-LEAKAGE HOLDOUT SETS")
    print("=" * 70)

    # 1.1 Singles Test
    singles_csv = FLAVRI_DIR / "data/quarantining/singles_test.csv"
    singles_ds = FlavriSinglesDataset(singles_csv)
    singles_loader = DataLoader(
        singles_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=SinglesCollator(device=device),
    )

    all_preds = []
    all_targets = []
    all_masks = []
    with torch.no_grad():
        for batch in singles_loader:
            graphs = batch["graphs"]
            targets = batch["targets"].cpu().numpy()
            masks = batch["masks"].cpu().numpy()

            _, preds = model.predict_singles(graphs)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets)
            all_masks.append(masks)

    preds_arr = np.concatenate(all_preds, axis=0)      # (N, 148)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 148)
    masks_arr = np.concatenate(all_masks, axis=0)      # (N, 148)

    # Taste channels (0..10)
    taste_preds = preds_arr[:, :NUM_TASTE]
    taste_targets = targets_arr[:, :NUM_TASTE]
    taste_masks = masks_arr[:, :NUM_TASTE]

    print("\n--- Taste Channels AUROC Breakdown (singles_test.csv) ---")
    taste_aurocs = {}
    for c, ch_name in enumerate(TASTE_CHANNELS):
        m_c = taste_masks[:, c] > 0
        if m_c.sum() >= 10:
            y_bin = (taste_targets[m_c, c] >= 0.5).astype(int)
            y_p = taste_preds[m_c, c]
            if len(np.unique(y_bin)) > 1:
                try:
                    auc = float(roc_auc_score(y_bin, y_p))
                    taste_aurocs[ch_name] = auc
                    print(f"  {ch_name:<20}: AUROC = {auc:.4f} (N={m_c.sum()}, Pos={y_bin.sum()})")
                except Exception as e:
                    pass
    overall_taste_macro = float(np.mean(list(taste_aurocs.values()))) if taste_aurocs else 0.0
    print(f"  => Taste Macro AUROC: {overall_taste_macro:.4f}")

    # Odor channels (11..147)
    odor_preds = preds_arr[:, NUM_TASTE:]
    odor_targets = targets_arr[:, NUM_TASTE:]
    odor_masks = masks_arr[:, NUM_TASTE:]

    odor_aurocs = []
    for c in range(NUM_ODOR):
        m_c = odor_masks[:, c] > 0
        if m_c.sum() >= 10:
            y_true = odor_targets[m_c, c]
            y_p = odor_preds[m_c, c]
            if len(np.unique(y_true)) > 1:
                try:
                    odor_aurocs.append(roc_auc_score(y_true, y_p))
                except Exception:
                    pass
    odor_macro = float(np.mean(odor_aurocs)) if odor_aurocs else 0.0

    odor_active = odor_masks > 0
    odor_micro = float(roc_auc_score(odor_targets[odor_active], odor_preds[odor_active])) if odor_active.sum() > 0 else 0.0

    top5_precs = []
    for i in range(len(odor_preds)):
        if odor_masks[i].sum() > 0:
            top5_idx = np.argsort(odor_preds[i])[-5:]
            top5_precs.append(odor_targets[i, top5_idx].sum() / 5.0)
    odor_top5_prec = float(np.mean(top5_precs)) if top5_precs else 0.0

    print("\n--- Odor Metrics Summary (singles_test.csv) ---")
    print(f"  Odor Macro AUROC: {odor_macro:.4f}")
    print(f"  Odor Micro AUROC: {odor_micro:.4f}")
    print(f"  Odor Top-5 Prec:  {odor_top5_prec:.4f}")

    # 1.2 Mixture Pairs Test
    pairs_csv = FLAVRI_DIR / "data/quarantining/pair_test.csv"
    pairs_ds = FlavriPairsDataset(pairs_csv)
    pairs_loader = DataLoader(
        pairs_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=PairsCollator(device=device),
    )

    records = {
        "half_cold": {"p_delta": [], "t_delta": [], "m": [], "p_blend": [], "t_blend": []},
        "double_cold": {"p_delta": [], "t_delta": [], "m": [], "p_blend": [], "t_blend": []},
    }

    with torch.no_grad():
        for batch in pairs_loader:
            unique_graphs = batch["unique_graphs"]
            idx_a = batch["idx_a"]
            idx_b = batch["idx_b"]
            true_deltas = batch["deltas"].cpu().numpy()
            masks = batch["masks"].cpu().numpy()
            cold_types = batch["cold_types"]

            pred_deltas, y_blend, v_a, v_b = model.predict_pairs_batch(unique_graphs, idx_a, idx_b)
            p_d_np = pred_deltas.detach().cpu().numpy()
            y_b_np = y_blend.detach().cpu().numpy()
            v_a_np = v_a.detach().cpu().numpy()
            v_b_np = v_b.detach().cpu().numpy()
            t_blend = np.clip(0.5 * v_a_np + 0.5 * v_b_np + true_deltas, 0.0, 1.0)

            for i, ct in enumerate(cold_types):
                key = ct if ct in records else "half_cold"
                records[key]["p_delta"].append(p_d_np[i])
                records[key]["t_delta"].append(true_deltas[i])
                records[key]["m"].append(masks[i])
                records[key]["p_blend"].append(y_b_np[i])
                records[key]["t_blend"].append(t_blend[i])

    print("\n--- Mixture Interaction Metrics Breakdown (pair_test.csv) ---")
    pair_metrics_table = {}
    for cohort in ["half_cold", "double_cold"]:
        p_d = np.array(records[cohort]["p_delta"])
        t_d = np.array(records[cohort]["t_delta"])
        m = np.array(records[cohort]["m"])
        p_b = np.array(records[cohort]["p_blend"])
        t_b = np.array(records[cohort]["t_blend"])

        act = m > 0
        p_act = p_d[act]
        t_act = t_d[act]
        p_b_act = p_b[act]
        t_b_act = t_b[act]

        r_val, _ = pearsonr(p_act, t_act) if len(p_act) > 1 else (0.0, 0.0)
        rho_val, _ = spearmanr(p_act, t_act) if len(p_act) > 1 else (0.0, 0.0)
        f1_val = f1_score((np.abs(t_act) >= 0.1).astype(int), (np.abs(p_act) >= 0.1).astype(int), zero_division=0)
        blend_mae = float(np.mean(np.abs(p_b_act - t_b_act)))
        delta_mae = float(np.mean(np.abs(p_act - t_act)))

        sig = np.abs(t_act) >= 0.05
        dir_acc = (np.sign(p_act[sig]) == np.sign(t_act[sig])).mean() if sig.sum() > 0 else 1.0

        pair_metrics_table[cohort] = {
            "pearson_r": r_val,
            "spearman_rho": rho_val,
            "synergy_f1": f1_val,
            "blend_mae": blend_mae,
            "delta_mae": delta_mae,
            "direction_acc": dir_acc,
        }

        print(f"\n[{cohort.upper()}] ({len(p_d)} pairs, {act.sum()} active descriptors):")
        print(f"  Pearson r:        {r_val:.4f}")
        print(f"  Spearman rho:     {rho_val:.4f}")
        print(f"  Synergy F1:       {f1_val:.4f}")
        print(f"  Blend MAE:        {blend_mae:.4f}")
        print(f"  Delta MAE:        {delta_mae:.4f}")
        print(f"  Direction Acc:    {dir_acc:.2%}")

    # =========================================================================
    # PART 2: Single Molecule Qualitative Inference
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: SINGLE MOLECULE QUALITATIVE PROFILING")
    print("=" * 70)

    test_molecules = [
        ("Ethylamine (CCN)", "CCN", "Primary aliphatic amine; sharp, ammoniacal, fishy odor with pungent/burning taste."),
        ("Vanillin", "O=Cc1ccc(O)c(OC)c1", "Classic sweet, vanilla, phenolic flavor reference."),
        ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "Xanthine alkaloid; quintessential bitter standard."),
        ("Menthol", "CC1CCC(C(C1)O)C(C)C", "TRPM8 agonist; cooling sensation and minty herbal aroma."),
        ("Capsaicin", "CC(C)/C=C/CCCCCC(=O)NCc1ccc(O)c(OC)c1", "TRPV1 agonist; extreme heat and pungency."),
        ("L-Glutamate (MSG)", "OC(=O)CCC(N)C(=O)O", "T1R1/T1R3 agonist; savory umami standard."),
        ("D-Limonene", "CC1=CCC(CC1)C(=C)C", "Terpene standard; citrus, orange, fresh aroma."),
        ("Benzaldehyde", "c1ccccc1C=O", "Aromatic aldehyde; almond, nutty, cherry notes."),
        ("Isoamyl acetate", "CC(=O)OCCC(C)C", "Ester; sweet banana and fruity aroma."),
    ]

    for name, smiles, desc in test_molecules:
        g_batch = mol2graph([smiles], device=device)
        with torch.no_grad():
            z, pred_probs = model.predict_singles(g_batch)
            p = pred_probs[0].cpu().numpy()

        taste_p = p[:NUM_TASTE]
        odor_p = p[NUM_TASTE:]

        # Sort odor descriptors
        top_odor_idx = np.argsort(odor_p)[::-1]

        print(f"\n>>> Molecule: {name} | SMILES: {smiles}")
        print(f"    Expected: {desc}")
        print("    [Predicted Taste Profile]:")
        for t_idx, t_name in enumerate(TASTE_CHANNELS):
            prob = taste_p[t_idx]
            bar = "#" * int(prob * 20)
            if prob >= 0.05:
                print(f"      {t_name:<18}: {prob:6.2%} |{bar:<20}|")

        print("    [Top 8 Predicted Odor Descriptors]:")
        for rank in range(8):
            o_idx = top_odor_idx[rank]
            o_name = ODOR_CHANNELS[o_idx]
            prob = odor_p[o_idx]
            bar = "#" * int(prob * 20)
            print(f"      {rank+1}. {o_name:<16}: {prob:6.2%} |{bar:<20}|")

    # =========================================================================
    # PART 3: Mixture Pairing & Synergy Profiling (Coffee + Chocolate)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: MIXTURE PAIRING & SYNERGY ANALYSIS")
    print("=" * 70)

    mixture_cases = [
        (
            "Coffee + Cocoa Roast Pairing",
            "2-Furfurylthiol (Coffee Aroma)", "SCC1=CC=CO1",
            "Tetramethylpyrazine (Roast Cocoa/Nutty)", "Cc1nc(C)c(C)nc1C"
        ),
        (
            "Smoky Coffee + Bitter Dark Chocolate Pairing",
            "Guaiacol (Smoky/Coffee)", "COc1ccccc1O",
            "Theobromine (Bitter Cocoa)", "Cn1cnc2c1c(=O)[nH]c(=O)n2C"
        ),
        (
            "Vanilla Sweetness + Fresh Citrus Twist",
            "Vanillin (Vanilla)", "O=Cc1ccc(O)c(OC)c1",
            "D-Limonene (Citrus)", "CC1=CCC(CC1)C(=C)C"
        ),
    ]

    for mix_name, name_a, smi_a, name_b, smi_b in mixture_cases:
        batch_ab = mol2graph([smi_a, smi_b], device=device)

        with torch.no_grad():
            z_all, v_all = model.predict_singles(batch_ab)
            z_a, z_b = z_all[0:1], z_all[1:2]
            v_a, v_b = v_all[0:1], v_all[1:2]
            delta = model.predict_synergy(z_a, z_b)
            w_a = torch.tensor([[0.5]], device=device)
            w_b = torch.tensor([[0.5]], device=device)
            y_blend = model.blend_profiles(v_a, v_b, delta, w_a, w_b)

        v_a_np = v_a[0].cpu().numpy()
        v_b_np = v_b[0].cpu().numpy()
        delta_np = delta[0].cpu().numpy()
        blend_np = y_blend[0].cpu().numpy()
        linear_np = 0.5 * v_a_np + 0.5 * v_b_np

        print(f"\n>>> Blend Case: {mix_name}")
        print(f"    Component A: {name_a} ({smi_a})")
        print(f"    Component B: {name_b} ({smi_b})")

        # Top synergetic enhancements (delta > 0)
        enh_idx = np.argsort(delta_np)[::-1]
        top_enh = [(ALL_CHANNELS[i], delta_np[i], linear_np[i], blend_np[i]) for i in enh_idx if delta_np[i] > 0.01][:5]

        # Top suppressions / masking (delta < 0)
        sup_idx = np.argsort(delta_np)
        top_sup = [(ALL_CHANNELS[i], delta_np[i], linear_np[i], blend_np[i]) for i in sup_idx if delta_np[i] < -0.01][:5]

        print("    [Top Synergistic Enhancements (Delta > 0)]:")
        if top_enh:
            for ch, d, lin, bl in top_enh:
                print(f"      + {ch:<16}: Delta = {d:+.4f} (Linear: {lin:5.2%} -> Blend: {bl:5.2%})")
        else:
            print("      (None > +0.01)")

        print("    [Top Masked / Suppressed Notes (Delta < 0)]:")
        if top_sup:
            for ch, d, lin, bl in top_sup:
                print(f"      - {ch:<16}: Delta = {d:+.4f} (Linear: {lin:5.2%} -> Blend: {bl:5.2%})")
        else:
            print("      (None < -0.01)")

        # Overall Blend Profile top notes
        top_blend_idx = np.argsort(blend_np)[::-1][:6]
        print("    [Resulting Dominant Sensory Notes in Mixture]:")
        for r, i in enumerate(top_blend_idx):
            ch = ALL_CHANNELS[i]
            print(f"      {r+1}. {ch:<16}: {blend_np[i]:6.2%} (A: {v_a_np[i]:.2%}, B: {v_b_np[i]:.2%}, Delta: {delta_np[i]:+.4f})")

if __name__ == "__main__":
    main()
