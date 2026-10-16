"""
data/quarantining/verify_integrity.py
Comprehensive verification script for multimodal merging and cold-start quarantining.
"""

import sys
import json
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from rdkit import Chem

from data.collection.odor_ingestion.constants import CANONICAL_ODOR_DESCRIPTORS

# File paths
COLLECTION_DIR = REPO_ROOT / "data" / "collection"
MULTIMODAL_SINGLES = COLLECTION_DIR / "flavri_multimodal_singles.csv"
MULTIMODAL_PAIRS = COLLECTION_DIR / "flavri_multimodal_pairs.csv"

QUARANTINE_DIR = REPO_ROOT / "data" / "quarantining"
SINGLES_TRAIN = QUARANTINE_DIR / "singles_train.csv"
SINGLES_TEST = QUARANTINE_DIR / "singles_test.csv"
PAIR_TRAIN = QUARANTINE_DIR / "pair_train.csv"
PAIR_TEST = QUARANTINE_DIR / "pair_test.csv"
SPLIT_META = QUARANTINE_DIR / "universe_split.json"


def verify_all():
    print("=" * 60)
    print("CHEMOSENSORY VERIFICATION & INTEGRITY SUITE")
    print("=" * 60)

    # 1. Load Files
    print("\n--- 1. Loading Datasets ---")
    df_ms = pd.read_csv(MULTIMODAL_SINGLES)
    df_mp = pd.read_csv(MULTIMODAL_PAIRS)
    df_st = pd.read_csv(SINGLES_TRAIN)
    df_ste = pd.read_csv(SINGLES_TEST)
    df_pt = pd.read_csv(PAIR_TRAIN)
    df_pte = pd.read_csv(PAIR_TEST)

    print(f"Multimodal Singles: {len(df_ms)} rows, {len(df_ms.columns)} cols (Expected: >=15,905 rows, 299 cols)")
    print(f"Multimodal Pairs:   {len(df_mp)} rows, {len(df_mp.columns)} cols (Expected: >=45,000 rows, 299 cols)")
    print(f"Singles Train:      {len(df_st)} rows, {len(df_st.columns)} cols")
    print(f"Singles Test:       {len(df_ste)} rows, {len(df_ste.columns)} cols")
    print(f"Pair Train:         {len(df_pt)} rows, {len(df_pt.columns)} cols")
    print(f"Pair Test:          {len(df_pte)} rows, {len(df_pte.columns)} cols (cold_type included: 300 cols)")

    assert len(df_ms) >= 15905, f"Expected at least 15905 singles rows, got {len(df_ms)}"
    assert len(df_ms.columns) == 299, f"Expected 299 singles cols, got {len(df_ms.columns)}"
    assert len(df_mp) >= 45000, f"Expected at least 45000 pairs rows, got {len(df_mp)}"
    assert len(df_mp.columns) == 299, f"Expected 299 pairs cols, got {len(df_mp.columns)}"
    assert len(df_st) + len(df_ste) == len(df_ms), "Singles partition does not sum to total singles!"
    assert len(df_pt) + len(df_pte) == len(df_mp), "Pairs partition does not sum to total pairs!"

    # 2. Channel Identification
    taste_y_cols = [c for c in df_ms.columns if c.startswith("y_taste_")]
    odor_y_cols = [c for c in df_ms.columns if c.startswith("y_") and not c.startswith("y_taste_")]
    taste_mask_cols = [c for c in df_ms.columns if c.startswith("mask_taste_")]
    odor_mask_cols = [c for c in df_ms.columns if c.startswith("mask_") and not c.startswith("mask_taste_")]

    assert len(taste_y_cols) == 11, f"Expected 11 taste y cols, got {len(taste_y_cols)}"
    assert len(odor_y_cols) == 137, f"Expected 137 odor y cols, got {len(odor_y_cols)}"
    assert len(taste_mask_cols) == 11, f"Expected 11 taste mask cols, got {len(taste_mask_cols)}"
    assert len(odor_mask_cols) == 137, f"Expected 137 odor mask cols, got {len(odor_mask_cols)}"

    # 3. Check SMILES Validity via RDKit
    print("\n--- 2. Checking RDKit SMILES Validity ---")
    all_smiles = set(df_ms["smiles"]).union(set(df_mp["smiles_a"])).union(set(df_mp["smiles_b"]))
    print(f"Total unique SMILES in universe: {len(all_smiles)}")
    invalid_smiles = [s for s in all_smiles if Chem.MolFromSmiles(s) is None]
    assert len(invalid_smiles) == 0, f"Found invalid SMILES: {invalid_smiles}"
    print("[PASS] 100% of SMILES parse validly in RDKit!")

    # 4. Check Lexicographical Ordering of Pairs
    print("\n--- 3. Checking Canonical Pair Ordering (smiles_a <= smiles_b) ---")
    assert (df_mp["smiles_a"] <= df_mp["smiles_b"]).all(), "Violated canonical pair ordering in multimodal pairs!"
    assert (df_pt["smiles_a"] <= df_pt["smiles_b"]).all(), "Violated canonical pair ordering in pair train!"
    assert (df_pte["smiles_a"] <= df_pte["smiles_b"]).all(), "Violated canonical pair ordering in pair test!"
    print("[PASS] 100% of pairs satisfy smiles_a <= smiles_b!")

    # 5. Check Zero-Leakage & Cold-Start Invariants
    print("\n--- 4. Checking Zero-Leakage & Cold-Start Invariants ---")
    assert SPLIT_META.exists(), f"Missing split metadata: {SPLIT_META}"
    with open(SPLIT_META) as f:
        split_meta = json.load(f)
    u_train = set(split_meta["u_train"])
    u_test = set(split_meta["u_test"])

    # Universe invariants
    assert len(u_train.intersection(u_test)) == 0, "U_train and U_test are not disjoint!"
    assert len(u_train) + len(u_test) == len(all_smiles), "U_train and U_test do not partition U!"

    train_mols = set(df_st["smiles"]).union(set(df_pt["smiles_a"])).union(set(df_pt["smiles_b"]))
    test_singles_mols = set(df_ste["smiles"])

    # Strict Zero-Leakage Assertion:
    # set(All molecules in singles_train and pair_train) & set(All molecules in singles_test) == empty set
    leakage = train_mols.intersection(test_singles_mols)
    assert len(leakage) == 0, f"Critical leakage detected between train and test singles: {leakage}"
    print(f"[PASS] Zero-Leakage Assertion holds: train_mols & test_singles_mols == empty set (0 overlap)")

    # 6. Pair Contamination Assertion: No molecule in pair_train.csv is in U_test
    pair_train_mols = set(df_pt["smiles_a"]).union(set(df_pt["smiles_b"]))
    pair_contamination = pair_train_mols.intersection(u_test)
    assert len(pair_contamination) == 0, f"Contamination in pair_train: {pair_contamination}"
    print(f"[PASS] Pair Contamination Assertion holds: 0 molecules in pair_train are in U_test!")

    # 7. Check Test Pair Cold Types
    print("\n--- 5. Checking Test Pair Diagnostic Categories ---")
    for _, row in df_pte.iterrows():
        a_cold = row["smiles_a"] in u_test
        b_cold = row["smiles_b"] in u_test
        assert a_cold or b_cold, "Test pair has neither constituent in U_test!"
        if a_cold and b_cold:
            assert row["cold_type"] == "double_cold", f"Expected double_cold, got {row['cold_type']}"
        else:
            assert row["cold_type"] == "half_cold", f"Expected half_cold, got {row['cold_type']}"
    print("[PASS] Test pair cold_type tags match exact constituent cold status!")

    # 8. Check Mask Integrity
    print("\n--- 6. Checking Mask Integrity ---")
    # Rule 1: Whenever mask = 0, target value is strictly 0.0
    for ch in taste_mask_cols + odor_mask_cols:
        target_col = "y_" + ch[len("mask_"):]
        bad = (df_ms[ch] == 0) & (df_ms[target_col] != 0.0)
        assert bad.sum() == 0, f"Singles mask integrity violated for {target_col}"

    # Rule 2: For pairs, whenever mask = 0, delta = 0.0
    taste_pair_masks = [c for c in df_mp.columns if c.startswith("mask_taste_")]
    odor_pair_masks = [c for c in df_mp.columns if c.startswith("mask_") and not c.startswith("mask_taste_")]
    for ch in taste_pair_masks + odor_pair_masks:
        delta_col = "delta_" + ch[len("mask_"):]
        bad_p = (df_mp[ch] == 0) & (df_mp[delta_col] != 0.0)
        assert bad_p.sum() == 0, f"Pairs mask integrity violated for {delta_col}"

    # Rule 3: For singles and pairs, whenever mask = 0, y/delta = 0.0 (enforces zero unassayed leakage)
    for ch in odor_mask_cols:
        y_col = "y_" + ch[len("mask_"):]
        bad_odor = (df_ms[ch] == 0) & (df_ms[y_col] != 0.0)
        assert bad_odor.sum() == 0, f"Odor mask integrity violated for {y_col}"

    # Rule 4: In odor pairs (where taste mask == 0), all 137 odor channels receive mask = 1
    odor_pair_rows = df_mp[df_mp[taste_pair_masks].sum(axis=1) == 0]
    assert (odor_pair_rows[odor_pair_masks] == 1).all().all(), "Not all odor channels are mask=1 in odor pairs!"

    print("[PASS] Mask integrity verified: whenever mask=0, value==0.0 (strictly preserved across taste and odor)!")

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY & PARTITION METRICS")
    print("=" * 60)
    print(f"Global Molecular Universe |U|:       {len(all_smiles)}")
    print(f"  - |U_train|:                       {len(u_train)} ({len(u_train)/len(all_smiles):.2%})")
    print(f"  - |U_test|:                        {len(u_test)} ({len(u_test)/len(all_smiles):.2%})")
    print(f"Multimodal Singles:                  {len(df_ms)} total rows")
    print(f"  - singles_train.csv:               {len(df_st)} rows ({len(df_st)/len(df_ms):.2%})")
    print(f"  - singles_test.csv:                {len(df_ste)} rows ({len(df_ste)/len(df_ms):.2%})")
    print(f"Multimodal Pairs:                    {len(df_mp)} total rows")
    print(f"  - pair_train.csv:                  {len(df_pt)} rows ({len(df_pt)/len(df_mp):.2%})")
    print(f"  - pair_test.csv:                   {len(df_pte)} rows ({len(df_pte)/len(df_mp):.2%})")
    print(f"    * half_cold:                     {(df_pte['cold_type'] == 'half_cold').sum()} rows")
    print(f"    * double_cold:                   {(df_pte['cold_type'] == 'double_cold').sum()} rows")
    print("=" * 60)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    verify_all()
