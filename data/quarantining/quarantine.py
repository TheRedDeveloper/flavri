"""
data/quarantining/quarantine.py
Perform strict 10% molecular cold-start partitioning across multimodal singles and pairs.
Guarantees zero train-test molecular leakage.
"""

import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

# Input file paths
COLLECTION_DIR = REPO_ROOT / "data" / "collection"
MULTIMODAL_SINGLES_CSV = COLLECTION_DIR / "flavri_multimodal_singles.csv"
MULTIMODAL_PAIRS_CSV = COLLECTION_DIR / "flavri_multimodal_pairs.csv"

# Output directory and file paths
QUARANTINE_DIR = REPO_ROOT / "data" / "quarantining"
SINGLES_TRAIN_CSV = QUARANTINE_DIR / "singles_train.csv"
SINGLES_TEST_CSV = QUARANTINE_DIR / "singles_test.csv"
PAIR_TRAIN_CSV = QUARANTINE_DIR / "pair_train.csv"
PAIR_TEST_CSV = QUARANTINE_DIR / "pair_test.csv"

PEPTIDE_SOURCES = {
    "TPDB_export",
    "TPDB_derivatives",
    "UmamiMeta_2024",
    "iUmami_TR",
    "iUmami_TS",
    "iUmami_Ext",
    "VirtuousUmami",
    "Sensoproteomics_TableS8",
}


def classify_molecule_strata(
    df_singles: pd.DataFrame, all_universe_smiles: set[str]
) -> dict[str, str]:
    """
    Classifies each molecule in universe U into stratified subgroups:
    - dual_flavorant: evaluated across both taste and odor
    - pure_odor: odor small molecule
    - taste_peptide: taste oligopeptide / polypeptide
    - taste_small_mol: non-peptide taste small molecule
    - pair_only: molecule appearing only in mixture pairs
    """
    taste_mask_cols = [c for c in df_singles.columns if c.startswith("mask_taste_")]
    odor_mask_cols = [c for c in df_singles.columns if c.startswith("mask_") and not c.startswith("mask_taste_")]

    singles_dict = {}
    for _, row in df_singles.iterrows():
        s = row["smiles"]
        has_taste = row[taste_mask_cols].sum() > 0
        has_odor = row[odor_mask_cols].sum() > 0
        src_str = str(row["sources"]) if pd.notna(row["sources"]) else ""
        src_set = set(src_str.split(";"))
        is_pep = bool(src_set.intersection(PEPTIDE_SOURCES))

        if has_taste and has_odor:
            stratum = "dual_flavorant"
        elif has_odor:
            stratum = "pure_odor"
        elif has_taste:
            if is_pep:
                stratum = "taste_peptide"
            else:
                stratum = "taste_small_mol"
        else:
            stratum = "pure_odor" if has_odor else "taste_small_mol"
        singles_dict[s] = stratum

    universe_strata = {}
    for s in all_universe_smiles:
        if s in singles_dict:
            universe_strata[s] = singles_dict[s]
        else:
            universe_strata[s] = "pair_only"

    return universe_strata


def partition_universe(
    universe_strata: dict[str, str], seed: int = 42, test_ratio: float = 0.10
) -> tuple[set[str], set[str]]:
    """
    Deterministically partitions universe U into U_train and U_test
    via stratified representative sampling.
    """
    rng = np.random.default_rng(seed)
    u_test = set()
    u_train = set()

    # Sort stratum names for consistent iteration
    all_strata = sorted(list(set(universe_strata.values())))

    print("\n--- Stratified Sampling Breakdown ---")
    for stratum_name in all_strata:
        mols = sorted([m for m, st in universe_strata.items() if st == stratum_name])
        n_total = len(mols)
        n_test = int(round(n_total * test_ratio))
        shuffled = rng.permutation(mols)
        test_mols = set(shuffled[:n_test])
        train_mols = set(shuffled[n_test:])

        u_test.update(test_mols)
        u_train.update(train_mols)

        print(
            f"Stratum '{stratum_name}': Total={n_total:5d} | "
            f"Test={len(test_mols):4d} ({len(test_mols)/n_total:6.2%}) | "
            f"Train={len(train_mols):5d} ({len(train_mols)/n_total:6.2%})"
        )

    # Invariants
    assert len(u_test.intersection(u_train)) == 0, "U_train and U_test must be strictly disjoint!"
    assert len(u_test) + len(u_train) == len(universe_strata), "Total partitioned molecules must match universe U!"

    return u_train, u_test


def main():
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Loading Merged Datasets ===")
    df_singles = pd.read_csv(MULTIMODAL_SINGLES_CSV)
    print(f"Loaded multimodal singles: {len(df_singles)} rows")

    df_pairs = pd.read_csv(MULTIMODAL_PAIRS_CSV)
    print(f"Loaded multimodal pairs: {len(df_pairs)} rows")

    # Define Global Universe U
    singles_smiles = set(df_singles["smiles"])
    pair_smiles = set(df_pairs["smiles_a"]).union(set(df_pairs["smiles_b"]))
    universe_u = sorted(list(singles_smiles.union(pair_smiles)))
    print(f"\nGlobal Molecular Universe U: {len(universe_u)} unique molecules")

    # Classify into strata
    strata = classify_molecule_strata(df_singles, set(universe_u))

    # Partition universe
    u_train, u_test = partition_universe(strata, seed=42, test_ratio=0.10)
    print(f"\nPartitioning Results:")
    print(f"  |U_test|  = {len(u_test)} ({len(u_test)/len(universe_u):.2%})")
    print(f"  |U_train| = {len(u_train)} ({len(u_train)/len(universe_u):.2%})")

    # Partition Singles
    print("\n=== Step 2: Partitioning Singles ===")
    df_singles_train = df_singles[df_singles["smiles"].isin(u_train)].copy()
    df_singles_test = df_singles[df_singles["smiles"].isin(u_test)].copy()

    print(f"Singles Train: {len(df_singles_train)} rows ({len(df_singles_train)/len(df_singles):.2%})")
    print(f"Singles Test:  {len(df_singles_test)} rows ({len(df_singles_test)/len(df_singles):.2%})")

    df_singles_train.to_csv(SINGLES_TRAIN_CSV, index=False)
    df_singles_test.to_csv(SINGLES_TEST_CSV, index=False)
    print(f"Saved: {SINGLES_TRAIN_CSV}")
    print(f"Saved: {SINGLES_TEST_CSV}")

    # Partition Pairs
    print("\n=== Step 3: Partitioning Pairs ===")
    in_train_a = df_pairs["smiles_a"].isin(u_train)
    in_train_b = df_pairs["smiles_b"].isin(u_train)
    in_test_a = df_pairs["smiles_a"].isin(u_test)
    in_test_b = df_pairs["smiles_b"].isin(u_test)

    train_mask = in_train_a & in_train_b
    test_mask = in_test_a | in_test_b

    df_pairs_train = df_pairs[train_mask].copy()
    df_pairs_test = df_pairs[test_mask].copy()

    # Assign cold_type diagnostic column to test pairs
    # 'half_cold': exactly one constituent is in U_test
    # 'double_cold': both constituents are in U_test
    double_cold_mask = in_test_a[test_mask] & in_test_b[test_mask]
    cold_types = np.where(double_cold_mask, "double_cold", "half_cold")
    df_pairs_test["cold_type"] = cold_types

    print(f"Pairs Train: {len(df_pairs_train)} rows ({len(df_pairs_train)/len(df_pairs):.2%})")
    print(f"Pairs Test:  {len(df_pairs_test)} rows ({len(df_pairs_test)/len(df_pairs):.2%})")
    print(f"  - half_cold:   {(cold_types == 'half_cold').sum()} rows")
    print(f"  - double_cold: {(cold_types == 'double_cold').sum()} rows")

    df_pairs_train.to_csv(PAIR_TRAIN_CSV, index=False)
    df_pairs_test.to_csv(PAIR_TEST_CSV, index=False)
    print(f"Saved: {PAIR_TRAIN_CSV}")
    print(f"Saved: {PAIR_TEST_CSV}")

    # Save partition metadata and SMILES sets for downstream evaluation / training
    split_meta_path = QUARANTINE_DIR / "universe_split.json"
    import json
    with open(split_meta_path, "w") as f:
        json.dump({
            "seed": 42,
            "test_ratio": 0.10,
            "total_universe_count": len(universe_u),
            "u_train_count": len(u_train),
            "u_test_count": len(u_test),
            "u_train": sorted(list(u_train)),
            "u_test": sorted(list(u_test)),
        }, f, indent=2)
    print(f"Saved: {split_meta_path}")

    print("\nQuarantining complete.")



if __name__ == "__main__":
    main()
