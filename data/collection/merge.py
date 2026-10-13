"""
data/collection/merge.py
Merge single-molecule and pairwise mixture datasets across taste and odor modalities
into unified multimodal benchmarks.
"""

import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

from data.collection.odor_ingestion.constants import CANONICAL_ODOR_DESCRIPTORS


# File paths
BASE_DIR = Path(__file__).resolve().parent
TASTE_SINGLES_CSV = BASE_DIR / "flavri_taste_singles.csv"
ODOR_SINGLES_CSV = BASE_DIR / "flavri_odor_singles.csv"
TASTE_PAIRS_CSV = BASE_DIR / "flavri_taste_pairs.csv"
ODOR_PAIRS_CSV = BASE_DIR / "flavri_odor_pairs.csv"

OUTPUT_SINGLES_CSV = BASE_DIR / "flavri_multimodal_singles.csv"
OUTPUT_PAIRS_CSV = BASE_DIR / "flavri_multimodal_pairs.csv"

# 11 Canonical Taste / Trigeminal Channels
RAW_TASTE_CHANNELS = [
    "sweet",
    "salty",
    "sour",
    "bitter",
    "umami",
    "kokumi",
    "astringent",
    "cooling",
    "heat",
    "pungency",
    "tingling",
]

# Naming convention:
# Taste channels are prefixed with 'taste_' (11 channels)
TASTE_CHANNELS = [f"taste_{t}" for t in RAW_TASTE_CHANNELS]

# Overlapping keys between taste and odor receive 'odor_' prefix,
# remaining 133 canonical odor descriptors remain unprefixed.
OVERLAP_KEYS = set(RAW_TASTE_CHANNELS).intersection(set(CANONICAL_ODOR_DESCRIPTORS))

# Map raw canonical odor descriptor to output column channel name
ODOR_DESCRIPTOR_MAP = {
    d: (f"odor_{d}" if d in OVERLAP_KEYS else d)
    for d in CANONICAL_ODOR_DESCRIPTORS
}
ODOR_CHANNELS = [ODOR_DESCRIPTOR_MAP[d] for d in CANONICAL_ODOR_DESCRIPTORS]

# Unified 148 channels: 11 taste + 137 odor
ALL_CHANNELS = TASTE_CHANNELS + ODOR_CHANNELS
assert len(ALL_CHANNELS) == 148, f"Expected 148 channels, got {len(ALL_CHANNELS)}"


def combine_sources(src_taste: str, src_odor: str) -> tuple[int, str]:
    """Combines and deduplicates semicolon-delimited source tags."""
    sources_set = []
    seen = set()

    def add_from_str(s):
        if pd.isna(s) or not str(s).strip():
            return
        for item in str(s).split(";"):
            item_clean = item.strip()
            if item_clean and item_clean not in seen:
                seen.add(item_clean)
                sources_set.append(item_clean)

    add_from_str(src_taste)
    add_from_str(src_odor)

    combined_str = ";".join(sources_set)
    return len(sources_set), combined_str


def merge_singles() -> pd.DataFrame:
    """
    Merges taste and odor single-molecule datasets into flavri_multimodal_singles.csv.
    Universe: Outer join on canonical smiles (15,905 expected rows).
    Columns (299): smiles, 148 y_<channel>, 148 mask_<channel>, num_sources, sources.
    """
    print(f"Reading taste singles from {TASTE_SINGLES_CSV}...")
    df_taste = pd.read_csv(TASTE_SINGLES_CSV)
    print(f"Loaded {len(df_taste)} taste singles.")

    print(f"Reading odor singles from {ODOR_SINGLES_CSV}...")
    df_odor = pd.read_csv(ODOR_SINGLES_CSV)
    print(f"Loaded {len(df_odor)} odor singles.")

    # Unique SMILES across both datasets
    all_smiles = sorted(list(set(df_taste["smiles"]).union(set(df_odor["smiles"]))))
    print(f"Unified single molecules universe size: {len(all_smiles)}")

    # Index by smiles for fast row lookup
    df_taste_indexed = df_taste.set_index("smiles")
    df_odor_indexed = df_odor.set_index("smiles")

    taste_smiles_set = set(df_taste_indexed.index)
    odor_smiles_set = set(df_odor_indexed.index)

    rows = []
    for s in all_smiles:
        row = {"smiles": s}
        in_taste = s in taste_smiles_set
        in_odor = s in odor_smiles_set

        # 1. Taste channels (11)
        if in_taste:
            t_row = df_taste_indexed.loc[s]
            for t_raw, t_col in zip(RAW_TASTE_CHANNELS, TASTE_CHANNELS):
                row[f"y_{t_col}"] = float(t_row[f"y_{t_raw}"])
                row[f"mask_{t_col}"] = int(t_row[f"mask_{t_raw}"])
            t_src = t_row["sources"]
        else:
            for t_col in TASTE_CHANNELS:
                row[f"y_{t_col}"] = 0.0
                row[f"mask_{t_col}"] = 0
            t_src = ""

        # 2. Odor channels (137)
        if in_odor:
            o_row = df_odor_indexed.loc[s]
            for d_raw, o_col in zip(CANONICAL_ODOR_DESCRIPTORS, ODOR_CHANNELS):
                row[f"y_{o_col}"] = float(o_row[f"y_{d_raw}"])
                row[f"mask_{o_col}"] = 1
            o_src = o_row["sources"]
        else:
            for o_col in ODOR_CHANNELS:
                row[f"y_{o_col}"] = 0.0
                row[f"mask_{o_col}"] = 0
            o_src = ""

        # 3. Sources tally
        n_src, comb_src = combine_sources(t_src, o_src)
        row["num_sources"] = n_src
        row["sources"] = comb_src

        rows.append(row)

    # Establish ordered column list: smiles + 148 y_ + 148 mask_ + num_sources + sources
    col_order = (
        ["smiles"]
        + [f"y_{ch}" for ch in ALL_CHANNELS]
        + [f"mask_{ch}" for ch in ALL_CHANNELS]
        + ["num_sources", "sources"]
    )
    df_merged_singles = pd.DataFrame(rows)[col_order]
    assert len(df_merged_singles.columns) == 299, f"Expected 299 columns, got {len(df_merged_singles.columns)}"
    return df_merged_singles


def merge_pairs() -> pd.DataFrame:
    """
    Merges taste and odor pair datasets into flavri_multimodal_pairs.csv.
    Universe: Union of taste pairs (685) and odor pairs (58,458) -> 59,143 expected rows.
    Columns (299): smiles_a, smiles_b, 148 delta_<channel>, 148 mask_<channel>, pair_source.
    """
    print(f"Reading taste pairs from {TASTE_PAIRS_CSV}...")
    df_taste = pd.read_csv(TASTE_PAIRS_CSV)
    print(f"Loaded {len(df_taste)} taste pairs.")

    print(f"Reading odor pairs from {ODOR_PAIRS_CSV}...")
    df_odor = pd.read_csv(ODOR_PAIRS_CSV)
    print(f"Loaded {len(df_odor)} odor pairs.")

    rows = []

    # 1. Process taste pairs
    for _, t_row in df_taste.iterrows():
        s_a, s_b = str(t_row["smiles_a"]), str(t_row["smiles_b"])
        if s_a > s_b:
            s_a, s_b = s_b, s_a
        row = {
            "smiles_a": s_a,
            "smiles_b": s_b,
        }
        # Taste channels
        for t_raw, t_col in zip(RAW_TASTE_CHANNELS, TASTE_CHANNELS):
            row[f"delta_{t_col}"] = float(t_row[f"delta_{t_raw}"])
            row[f"mask_{t_col}"] = int(t_row[f"mask_{t_raw}"])
        # Odor channels unmeasured
        for o_col in ODOR_CHANNELS:
            row[f"delta_{o_col}"] = 0.0
            row[f"mask_{o_col}"] = 0
        row["pair_source"] = str(t_row["pair_source"])
        rows.append(row)

    # 2. Process odor pairs
    for _, o_row in df_odor.iterrows():
        s_a, s_b = str(o_row["smiles_a"]), str(o_row["smiles_b"])
        if s_a > s_b:
            s_a, s_b = s_b, s_a
        row = {
            "smiles_a": s_a,
            "smiles_b": s_b,
        }
        # Taste channels unmeasured
        for t_col in TASTE_CHANNELS:
            row[f"delta_{t_col}"] = 0.0
            row[f"mask_{t_col}"] = 0
        # Odor channels: mask = 1 across all 137 channels (including delta = 0.0)
        for d_raw, o_col in zip(CANONICAL_ODOR_DESCRIPTORS, ODOR_CHANNELS):
            row[f"delta_{o_col}"] = float(o_row[f"delta_{d_raw}"])
            row[f"mask_{o_col}"] = 1
        row["pair_source"] = str(o_row["pair_source"])
        rows.append(row)

    # Establish ordered column list: smiles_a, smiles_b + 148 delta_ + 148 mask_ + pair_source
    col_order = (
        ["smiles_a", "smiles_b"]
        + [f"delta_{ch}" for ch in ALL_CHANNELS]
        + [f"mask_{ch}" for ch in ALL_CHANNELS]
        + ["pair_source"]
    )
    df_merged_pairs = pd.DataFrame(rows)[col_order]

    # Deduplicate if any exact identical pair existed
    initial_count = len(df_merged_pairs)
    df_merged_pairs = df_merged_pairs.drop_duplicates(subset=["smiles_a", "smiles_b"])
    if len(df_merged_pairs) < initial_count:
        print(f"Deduplicated pairs: {initial_count} -> {len(df_merged_pairs)}")

    assert len(df_merged_pairs.columns) == 299, f"Expected 299 columns, got {len(df_merged_pairs.columns)}"
    return df_merged_pairs


def main():
    print("=== Step 1: Merging Multimodal Singles ===")
    df_singles = merge_singles()
    print(f"Saving merged singles to {OUTPUT_SINGLES_CSV} ({len(df_singles)} rows, {len(df_singles.columns)} cols)...")
    df_singles.to_csv(OUTPUT_SINGLES_CSV, index=False)
    print("Multimodal singles successfully written.")

    print("\n=== Step 2: Merging Multimodal Pairs ===")
    df_pairs = merge_pairs()
    print(f"Saving merged pairs to {OUTPUT_PAIRS_CSV} ({len(df_pairs)} rows, {len(df_pairs.columns)} cols)...")
    df_pairs.to_csv(OUTPUT_PAIRS_CSV, index=False)
    print("Multimodal pairs successfully written.")


if __name__ == "__main__":
    main()
