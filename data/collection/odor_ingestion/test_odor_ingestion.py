"""
test_odor_ingestion.py
Comprehensive test and quality assurance suite for FLAVRI_ODOR datasets.
"""

import unittest
from pathlib import Path
import pandas as pd
from rdkit import Chem

from odor_ingestion.constants import (
    OUTPUT_SINGLES_CSV,
    OUTPUT_PAIRS_CSV,
    CANONICAL_ODOR_DESCRIPTORS,
    RAW_BP_DIR,
    SYNONYM_MAP,
)


class TestFlavriOdorDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nLoading FLAVRI_ODOR datasets for validation...")
        cls.df_singles = pd.read_csv(OUTPUT_SINGLES_CSV)
        cls.df_pairs = pd.read_csv(OUTPUT_PAIRS_CSV)
        cls.singles_smiles = set(cls.df_singles["smiles"])

        # Identify active and preserved channels
        with open(RAW_BP_DIR.parent / "bp_74.txt") as f:
            bp_74 = [l.strip() for l in f if l.strip()]
        cls.active_channels = set()
        for tag in bp_74:
            if tag in SYNONYM_MAP:
                cls.active_channels.add(SYNONYM_MAP[tag])
            elif tag in CANONICAL_ODOR_DESCRIPTORS:
                cls.active_channels.add(tag)
        cls.preserved_channels = [d for d in CANONICAL_ODOR_DESCRIPTORS if d not in cls.active_channels]

    # ==========================================
    # 1. Singles Table Tests
    # ==========================================
    def test_singles_file_exists_and_non_empty(self):
        self.assertTrue(OUTPUT_SINGLES_CSV.exists())
        self.assertGreater(len(self.df_singles), 4500)

    def test_singles_column_count_and_schema(self):
        expected_cols = ["smiles"]
        for d in CANONICAL_ODOR_DESCRIPTORS:
            expected_cols.append(f"y_{d}")
        for d in CANONICAL_ODOR_DESCRIPTORS:
            expected_cols.append(f"mask_{d}")
        expected_cols.extend(["num_sources", "sources"])

        self.assertEqual(len(self.df_singles.columns), 277)
        self.assertEqual(list(self.df_singles.columns), expected_cols)

    def test_singles_rdkit_parseability(self):
        for smi in self.df_singles["smiles"]:
            mol = Chem.MolFromSmiles(smi)
            self.assertIsNotNone(mol, f"Unparseable SMILES in singles: {smi}")
            can = Chem.MolToSmiles(mol)
            self.assertEqual(smi, can, f"SMILES not canonical in singles: {smi} vs {can}")

    def test_singles_uniqueness(self):
        self.assertEqual(self.df_singles["smiles"].nunique(), len(self.df_singles))

    def test_singles_no_nans(self):
        nan_count = self.df_singles.isnull().sum().sum()
        self.assertEqual(nan_count, 0, f"Found {nan_count} NaNs in singles table")

    def test_singles_potency_and_mask_ranges(self):
        for d in CANONICAL_ODOR_DESCRIPTORS:
            y_col = f"y_{d}"
            mask_col = f"mask_{d}"
            self.assertTrue(((self.df_singles[y_col] >= 0.0) & (self.df_singles[y_col] <= 1.0)).all())
            self.assertTrue((self.df_singles[mask_col].isin([0, 1])).all())

    def test_singles_sources(self):
        allowed_base_sources = {"GoodScents", "Leffingwell", "MMD", "OlfactionBase"}
        for s in self.df_singles["sources"].unique():
            parts = s.split(";")
            self.assertTrue(set(parts).issubset(allowed_base_sources), f"Unexpected source: {s}")
        self.assertTrue((self.df_singles["num_sources"] >= 1).all())

    # ==========================================
    # 2. Pairs Table Tests
    # ==========================================
    def test_pairs_file_exists_and_non_empty(self):
        self.assertTrue(OUTPUT_PAIRS_CSV.exists())
        self.assertGreater(len(self.df_pairs), 45000)

    def test_pairs_column_count_and_schema(self):
        expected_cols = ["smiles_a", "smiles_b"]
        for d in CANONICAL_ODOR_DESCRIPTORS:
            expected_cols.append(f"delta_{d}")
        expected_cols.append("pair_source")

        self.assertEqual(len(self.df_pairs.columns), 140)
        self.assertEqual(list(self.df_pairs.columns), expected_cols)

    def test_pairs_no_nans(self):
        nan_count = self.df_pairs.isnull().sum().sum()
        self.assertEqual(nan_count, 0, f"Found {nan_count} NaNs in pairs table")

    def test_pairs_commutative_symmetry(self):
        violations = (self.df_pairs["smiles_a"] >= self.df_pairs["smiles_b"]).sum()
        self.assertEqual(violations, 0, f"Found {violations} commutative symmetry violations (a >= b)")

    def test_pairs_constituents_exist_in_singles(self):
        missing_a = set(self.df_pairs["smiles_a"]) - self.singles_smiles
        missing_b = set(self.df_pairs["smiles_b"]) - self.singles_smiles
        self.assertEqual(len(missing_a), 0, f"Found smiles_a not in singles: {len(missing_a)}")
        self.assertEqual(len(missing_b), 0, f"Found smiles_b not in singles: {len(missing_b)}")

    def test_pairs_source_constant(self):
        self.assertTrue((self.df_pairs["pair_source"] == "BlendedPairs").all())

    def test_pairs_delta_ranges(self):
        delta_cols = [f"delta_{d}" for d in CANONICAL_ODOR_DESCRIPTORS]
        min_val = self.df_pairs[delta_cols].min().min()
        max_val = self.df_pairs[delta_cols].max().max()
        self.assertGreaterEqual(min_val, -1.0)
        self.assertLessEqual(max_val, 1.0)

    def test_pairs_neutrality_anchors_strictly_zero(self):
        self.assertEqual(len(self.preserved_channels), 75)
        for d in self.preserved_channels:
            col = f"delta_{d}"
            non_zeros = (self.df_pairs[col] != 0.0).sum()
            self.assertEqual(non_zeros, 0, f"Preserved channel {d} has {non_zeros} non-zero values!")


if __name__ == "__main__":
    unittest.main()
