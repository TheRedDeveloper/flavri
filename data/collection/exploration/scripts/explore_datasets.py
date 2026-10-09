"""
explore_datasets.py
Exploratory data analysis of downloaded taste, chemesthesis, and mixture datasets.
"""

import os
import glob
from pathlib import Path
import pandas as pd
from rdkit import Chem

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def explore_chemtastesdb():
    print("\n--- 1. Exploring ChemTastesDB ---")
    fpath = RAW_DIR / "chemtastesdb" / "ChemTastesDB_database.xlsx"
    if not fpath.exists():
        print(f"File not found: {fpath}")
        return
    
    xl = pd.ExcelFile(fpath)
    print(f"Sheets: {xl.sheet_names}")
    df = xl.parse(xl.sheet_names[0])
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Taste classes column
    taste_col = [c for c in df.columns if "class" in c.lower() or "taste" in c.lower()]
    print(f"Candidate taste columns: {taste_col}")
    for tc in taste_col:
        print(f"Distribution of {tc}:")
        print(df[tc].value_counts().head(15))
        print("-" * 30)

    # Validate SMILES with RDKit
    smiles_col = [c for c in df.columns if "smiles" in c.lower()][0]
    valid_mols = 0
    disconnected_mols = 0
    for s in df[smiles_col].dropna():
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            valid_mols += 1
            if "." in str(s):
                disconnected_mols += 1
    print(f"Total non-null SMILES: {df[smiles_col].dropna().shape[0]}")
    print(f"Valid RDKit mols: {valid_mols}")
    print(f"Disconnected (salts/ionic) mols: {disconnected_mols}")

    # Save summary
    df.head(5).to_csv(ANALYSIS_DIR / "chemtastesdb_head.csv", index=False)

def explore_bittergnn():
    print("\n--- 2. Exploring BitterGNN Datasets ---")
    bdir = RAW_DIR / "peptides_heyigacu" / "bittergnn" / "dataset"
    if not bdir.exists():
        print(f"Directory not found: {bdir}")
        return
    files = list(bdir.rglob("*.csv")) + list(bdir.rglob("*.xlsx"))
    print(f"Found {len(files)} files in BitterGNN dataset")
    for f in files:
        try:
            df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_excel(f)
            print(f"  [{f.name}] Shape: {df.shape}, Cols: {list(df.columns)[:5]}")
        except Exception as e:
            print(f"  [{f.name}] Error reading: {e}")

def explore_kokumipd():
    print("\n--- 3. Exploring KokumiPD ---")
    kdir = RAW_DIR / "peptides_heyigacu" / "kokumipd"
    if not kdir.exists():
        return
    files = list(kdir.rglob("*.csv")) + list(kdir.rglob("*.txt"))
    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"  [{f.name}] Shape: {df.shape}, Cols: {list(df.columns)}")
            print(df.head(3))
        except Exception as e:
            print(f"  [{f.name}] Error reading: {e}")

def explore_umamimeta():
    print("\n--- 4. Exploring UmamiMeta ---")
    udir = RAW_DIR / "peptides_heyigacu" / "umamimeta"
    if not udir.exists():
        return
    files = list(udir.rglob("*.csv"))
    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"  [{f.name}] Shape: {df.shape}, Cols: {list(df.columns)[:6]}")
            if "sequence" in [c.lower() for c in df.columns] or "smiles" in [c.lower() for c in df.columns]:
                print(df.head(2))
        except Exception as e:
            print(f"  [{f.name}] Error reading: {e}")

def explore_virtuous_umami():
    print("\n--- 5. Exploring VirtuousUmami ---")
    vdir = RAW_DIR / "virtuous_umami" / "data"
    if not vdir.exists():
        return
    for fname in ["Umami_training_dataset.csv", "Umami_testing_dataset.csv"]:
        fpath = vdir / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            print(f"  [{fname}] Shape: {df.shape}, Cols: {list(df.columns)[:6]}")
            print(f"  Class distribution:\n{df.iloc[:, -1].value_counts()}")

def explore_cooling_trpm8():
    print("\n--- 6. Exploring TRPM8 Cooling Library ---")
    cpath = RAW_DIR / "cooling_trpm8" / "journal.pone.0288441.s001.xlsx"
    if not cpath.exists():
        return
    xl = pd.ExcelFile(cpath)
    print(f"Sheets: {xl.sheet_names}")
    for s in xl.sheet_names:
        df = xl.parse(s)
        print(f"  Sheet [{s}] Shape: {df.shape}, Cols: {list(df.columns)[:8]}")
        print(df.head(2))

def explore_psychophysical_mixtures():
    print("\n--- 7. Exploring Psychophysical Mixture Benchmarks ---" )
    pdir = RAW_DIR / "psychophysical_mixtures"
    if not pdir.exists():
        return
    for f in pdir.glob("*.csv"):
        df = pd.read_csv(f)
        print(f"  [{f.name}] Shape: {df.shape}, Cols: {list(df.columns)}")
        print(df.head(2))

def main():
    explore_chemtastesdb()
    explore_bittergnn()
    explore_kokumipd()
    explore_umamimeta()
    explore_virtuous_umami()
    explore_cooling_trpm8()
    explore_psychophysical_mixtures()

if __name__ == "__main__":
    main()
