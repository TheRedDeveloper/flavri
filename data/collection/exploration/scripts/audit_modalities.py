"""
audit_modalities.py
Systematic analysis of columns, sizes, identifiers, and modalities across all downloaded datasets.
"""

from pathlib import Path
import pandas as pd
from rdkit import Chem

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
ANALYSIS_DIR = BASE_DIR / "analysis"

def audit_chemtastesdb():
    print("=== 1. ChemTastesDB Audit ===")
    fpath = RAW_DIR / "chemtastesdb" / "ChemTastesDB_database.xlsx"
    xl = pd.ExcelFile(fpath)
    df = xl.parse(xl.sheet_names[0])
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check classes
    for col in ["Class", "Class_detailed", "Taste"]:
        if col in df.columns:
            print(f"\nValue counts for '{col}':")
            print(df[col].value_counts())

    # Check SMILES validity
    smiles_col = [c for c in df.columns if "smiles" in c.lower()][0]
    valid_count = 0
    invalid_examples = []
    salts_count = 0
    for idx, s in enumerate(df[smiles_col]):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            valid_count += 1
            if "." in str(s):
                salts_count += 1
        else:
            if len(invalid_examples) < 5:
                invalid_examples.append(str(s))
    print(f"Valid SMILES: {valid_count}/{len(df)} ({valid_count/len(df):.1%})")
    print(f"Salts/Disconnected entries ('.'): {salts_count}")
    if invalid_examples:
        print(f"Sample invalid: {invalid_examples}")

def audit_peptides_heyigacu():
    print("\n=== 2. Peptides Audit (Heyigacu Lab: UmamiMeta, KokumiPD, BitterGNN, etc.) ===")
    pdir = RAW_DIR / "peptides_heyigacu"
    
    # 2.1 UmamiMeta
    umm_path = pdir / "umamimeta" / "code" / "dock" / "ummpep2024.csv"
    if umm_path.exists():
        df_umm = pd.read_csv(umm_path)
        print(f"UmamiMeta (ummpep2024.csv): {len(df_umm)} entries")
        print(f"Cols: {list(df_umm.columns)}")
        print(df_umm.head(3))
        
    # 2.2 KokumiPD
    kokumi_path = pdir / "kokumipd" / "test_smiles.csv"
    if kokumi_path.exists():
        df_k = pd.read_csv(kokumi_path)
        print(f"\nKokumiPD (test_smiles.csv): {len(df_k)} entries")
        print(f"Cols: {list(df_k.columns)}")
        print(df_k.head(3))

    # 2.3 BitterGNN
    bitter_path = pdir / "bittergnn" / "dataset" / "taste_dataset"
    if bitter_path.exists():
        print("\nBitterGNN taste_dataset files:")
        for bf in bitter_path.glob("*.csv"):
            df_b = pd.read_csv(bf)
            print(f"  {bf.name}: {len(df_b)} entries | Cols: {list(df_b.columns)[:5]}")
            if "taste" in [c.lower() for c in df_b.columns] or "label" in [c.lower() for c in df_b.columns]:
                tcol = [c for c in df_b.columns if c.lower() in ["taste", "label"]][0]
                print(f"    Distribution: {dict(df_b[tcol].value_counts().head(5))}")

def audit_virtuous_umami():
    print("\n=== 3. VirtuousUmami Audit ===")
    vdir = RAW_DIR / "virtuous_umami" / "data"
    train_f = vdir / "Umami_training_dataset.csv"
    test_f = vdir / "Umami_testing_dataset.csv"
    if train_f.exists() and test_f.exists():
        df_tr = pd.read_csv(train_f)
        df_te = pd.read_csv(test_f)
        print(f"Train: {len(df_tr)} rows | Test: {len(df_te)} rows | Total: {len(df_tr)+len(df_te)}")
        print(f"Cols: {list(df_tr.columns)[:6]} ...")
        tcol = df_tr.columns[-1]
        print(f"Train class dist ({tcol}): {dict(df_tr[tcol].value_counts())}")
        print(f"Test class dist ({tcol}): {dict(df_te[tcol].value_counts())}")

def audit_saltymeta():
    print("\n=== 4. SaltyMeta Audit ===")
    sdir = RAW_DIR / "saltymeta"
    sample_f = sdir / "sample_sequences.csv"
    if sample_f.exists():
        df_s = pd.read_csv(sample_f)
        print(f"Sample sequences: {len(df_s)} entries")
        print(df_s.head(5))
        
def audit_jecfa():
    print("\n=== 5. JECFA Regulatory Database Audit ===")
    jdir = RAW_DIR / "jecfa_db"
    data_files = list(jdir.rglob("*.csv")) + list(jdir.rglob("*.rda")) + list(jdir.rglob("*.rds"))
    print(f"Found {len(data_files)} data files in JECFA repo:")
    for f in data_files[:10]:
        print(f"  {f.name} ({f.stat().st_size} bytes)")

def audit_mixtures():
    print("\n=== 6. Psychophysical Mixture Benchmarks Audit ===")
    pdir = RAW_DIR / "psychophysical_mixtures"
    for f in pdir.glob("*.csv"):
        df = pd.read_csv(f)
        print(f"  [{f.name}] Rows: {len(df)}, Cols: {list(df.columns)}")

if __name__ == "__main__":
    audit_chemtastesdb()
    audit_peptides_heyigacu()
    audit_virtuous_umami()
    audit_saltymeta()
    audit_jecfa()
    audit_mixtures()
