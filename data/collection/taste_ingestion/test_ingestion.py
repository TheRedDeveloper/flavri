"""
test_ingestion.py
Verification and test suite for FLAVRI taste dataset ingestion.
Validates:
1. File existence and schema
2. 100% RDKit molecule parsing for SMILES
3. Label bounds: Singles y in [0.0, 1.0]; Pairs delta in [-1.0, 1.0]
4. Mask bounds: mask in {0, 1}
5. Commutative symmetry: smiles_a < smiles_b
6. Representation of all 11 sensory channels
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem

from utils import CHANNELS

DATA_COLLECTION = Path(__file__).resolve().parent.parent


def test_singles_dataset():
    print("=== Testing FLAVRI_TASTE_SINGLES ===")
    path = DATA_COLLECTION / "flavri_taste_singles.csv"
    assert path.exists(), f"File does not exist: {path}"

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} singles records.")
    assert len(df) > 10000, f"Expected >10,000 compounds, found {len(df)}"

    # Schema checks
    assert "smiles" in df.columns, "Missing 'smiles' column"
    for ch in CHANNELS:
        assert f"y_{ch}" in df.columns, f"Missing y_{ch}"
        assert f"mask_{ch}" in df.columns, f"Missing mask_{ch}"

    # Range and mask checks
    for ch in CHANNELS:
        y_col = df[f"y_{ch}"]
        m_col = df[f"mask_{ch}"]
        assert y_col.min() >= 0.0, f"Negative value in y_{ch}: {y_col.min()}"
        assert y_col.max() <= 1.0, f"Value > 1.0 in y_{ch}: {y_col.max()}"
        assert set(m_col.unique()).issubset({0, 1}), f"Non-binary mask in mask_{ch}"

    # RDKit validation
    print("Validating RDKit SMILES parsing on all singles...")
    invalid_smiles = 0
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles += 1
    assert invalid_smiles == 0, f"Found {invalid_smiles} invalid SMILES in singles!"

    print("All singles checks passed successfully!\n")


def test_pairs_dataset():
    print("=== Testing FLAVRI_TASTE_PAIRS ===")
    path = DATA_COLLECTION / "flavri_taste_pairs.csv"
    assert path.exists(), f"File does not exist: {path}"

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} pairs records.")
    assert len(df) > 300, f"Expected >300 pairs, found {len(df)}"

    # Schema checks
    assert "smiles_a" in df.columns, "Missing 'smiles_a'"
    assert "smiles_b" in df.columns, "Missing 'smiles_b'"
    for ch in CHANNELS:
        assert f"delta_{ch}" in df.columns, f"Missing delta_{ch}"
        assert f"mask_{ch}" in df.columns, f"Missing mask_{ch}"

    # Range and mask checks
    for ch in CHANNELS:
        d_col = df[f"delta_{ch}"]
        m_col = df[f"mask_{ch}"]
        assert d_col.min() >= -1.0, f"Delta < -1.0 in delta_{ch}: {d_col.min()}"
        assert d_col.max() <= 1.0, f"Delta > 1.0 in delta_{ch}: {d_col.max()}"
        assert set(m_col.unique()).issubset({0, 1}), f"Non-binary mask in mask_{ch}"

    # Commutative order check
    print("Validating commutative canonical order smiles_a < smiles_b...")
    for _, row in df.iterrows():
        a = row["smiles_a"]
        b = row["smiles_b"]
        assert a < b, f"Order violation: {a} is not < {b}"

    # RDKit validation
    print("Validating RDKit SMILES parsing on all pairs...")
    invalid_pairs = 0
    for smi in list(df["smiles_a"]) + list(df["smiles_b"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_pairs += 1
    assert invalid_pairs == 0, f"Found {invalid_pairs} invalid SMILES in pairs!"

    print("All pairs checks passed successfully!\n")


if __name__ == "__main__":
    test_singles_dataset()
    test_pairs_dataset()
    print(">>> 100% OF INGESTION TESTS PASSED! <<<")
