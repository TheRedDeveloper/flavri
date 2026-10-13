"""
parse_odor_pairs.py
Constructs the canonical flavri_odor_pairs.csv master table without redundant ratios or masks.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from rdkit import Chem

from odor_ingestion.constants import (
    RAW_BP_DIR,
    OUTPUT_SINGLES_CSV,
    OUTPUT_PAIRS_CSV,
    CANONICAL_ODOR_DESCRIPTORS,
    SYNONYM_MAP,
    DISCARD_PHANTOMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("parse_odor_pairs")


def canonicalize_smiles(s: str) -> str:
    """Standardize SMILES using RDKit."""
    m = Chem.MolFromSmiles(s.strip())
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def parse_odor_pairs() -> pd.DataFrame:
    logger.info(f"Loading single odorants baseline from: {OUTPUT_SINGLES_CSV}")
    if not OUTPUT_SINGLES_CSV.exists():
        raise FileNotFoundError(f"Single odorants file not found at {OUTPUT_SINGLES_CSV}")
    
    df_singles = pd.read_csv(OUTPUT_SINGLES_CSV)
    logger.info(f"Loaded {len(df_singles)} single odorant baselines.")

    singles_dict = {}
    for _, row in df_singles.iterrows():
        smi = row["smiles"]
        potencies = {d: float(row[f"y_{d}"]) for d in CANONICAL_ODOR_DESCRIPTORS}
        singles_dict[smi] = potencies

    active_bp_channels = set()
    with open(RAW_BP_DIR.parent / "bp_74.txt") as f:
        bp_74 = [line.strip() for line in f if line.strip()]

    for tag in bp_74:
        if tag in SYNONYM_MAP:
            active_bp_channels.add(SYNONYM_MAP[tag])
        elif tag in CANONICAL_ODOR_DESCRIPTORS:
            active_bp_channels.add(tag)

    logger.info(f"Active BP interaction channels: {len(active_bp_channels)}")
    logger.info(f"Preserved neutrality anchor channels: {len(CANONICAL_ODOR_DESCRIPTORS) - len(active_bp_channels)}")

    bp_fold1 = RAW_BP_DIR / "fold1"
    split_files = [bp_fold1 / "train.csv", bp_fold1 / "valid.csv", bp_fold1 / "test.csv"]
    
    dfs = []
    for sf in split_files:
        if not sf.exists():
            raise FileNotFoundError(f"BP split file not found: {sf}")
        logger.info(f"Loading {sf.name}...")
        dfs.append(pd.read_csv(sf))
    df_raw_bp = pd.concat(dfs, axis=0)
    logger.info(f"Total raw BP rows: {len(df_raw_bp)}")

    raw_bp_cols = [c for c in df_raw_bp.columns if c != "smiles"]
    pair_annotations = {}
    skipped_unparseable = 0
    skipped_self_pair = 0
    skipped_missing_singles = 0

    for _, row in df_raw_bp.iterrows():
        s_pair = row["smiles"]
        parts = s_pair.split(";")
        if len(parts) != 2:
            skipped_unparseable += 1
            continue

        c1 = canonicalize_smiles(parts[0])
        c2 = canonicalize_smiles(parts[1])
        if c1 is None or c2 is None:
            skipped_unparseable += 1
            continue

        if c1 == c2:
            skipped_self_pair += 1
            continue

        # Enforce strict commutative ordering: a < b
        if c1 > c2:
            s_a, s_b = c2, c1
        else:
            s_a, s_b = c1, c2

        if s_a not in singles_dict or s_b not in singles_dict:
            skipped_missing_singles += 1
            continue

        pair_key = (s_a, s_b)
        if pair_key not in pair_annotations:
            pair_annotations[pair_key] = {d: 0.0 for d in CANONICAL_ODOR_DESCRIPTORS}

        for col in raw_bp_cols:
            val = float(row[col])
            if val > 0.0:
                if col in SYNONYM_MAP:
                    target = SYNONYM_MAP[col]
                    pair_annotations[pair_key][target] = 1.0
                elif col in CANONICAL_ODOR_DESCRIPTORS:
                    pair_annotations[pair_key][col] = 1.0

    logger.info(f"Unique valid canonical pairs: {len(pair_annotations)}")

    records = []
    for (s_a, s_b), blend_targets in pair_annotations.items():
        base_a = singles_dict[s_a]
        base_b = singles_dict[s_b]

        record = {
            "smiles_a": s_a,
            "smiles_b": s_b,
        }

        for desc in CANONICAL_ODOR_DESCRIPTORS:
            if desc in active_bp_channels:
                y_blend = blend_targets[desc]
                y_a = base_a[desc]
                y_b = base_b[desc]
                delta = y_blend - 0.5 * (y_a + y_b)
                delta = max(-1.0, min(1.0, round(delta, 6)))
                record[f"delta_{desc}"] = delta
            else:
                # Additive neutrality anchor
                record[f"delta_{desc}"] = 0.0

        record["pair_source"] = "BlendedPairs"
        records.append(record)

    df_pairs = pd.DataFrame(records)
    df_pairs = df_pairs.sort_values(["smiles_a", "smiles_b"]).reset_index(drop=True)

    # Order columns strictly: smiles_a, smiles_b, delta_desc1, ..., delta_desc137, pair_source
    ordered_cols = ["smiles_a", "smiles_b"]
    for d in CANONICAL_ODOR_DESCRIPTORS:
        ordered_cols.append(f"delta_{d}")
    ordered_cols.append("pair_source")

    df_pairs = df_pairs[ordered_cols]
    OUTPUT_PAIRS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(OUTPUT_PAIRS_CSV, index=False)
    logger.info(f"Successfully saved {len(df_pairs)} mixture pairs (140 columns) to {OUTPUT_PAIRS_CSV}")
    return df_pairs


if __name__ == "__main__":
    parse_odor_pairs()
