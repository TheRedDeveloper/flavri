"""
fetch_chembl_bioactivity_samples.py
Download verified ChEMBL bioactivity records (SMILES, EC50/IC50/Ki) for human sensory channels:
- TRPM8 (Cooling)
- TRPV1 (Heat)
- TRPA1 (Pungency)
- CaSR (Kokumi)
"""

import requests
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = BASE_DIR / "raw" / "chembl_receptors"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "TRPM8_cooling": "CHEMBL1075319",
    "TRPV1_heat": "CHEMBL4794",
    "TRPA1_pungency": "CHEMBL6007",
    "CaSR_kokumi": "CHEMBL1878"
}

def fetch_target_bioactivities(label: str, chembl_id: str, limit: int = 250):
    print(f"Fetching ChEMBL bioactivities for {label} ({chembl_id})...")
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={chembl_id}&standard_type__in=IC50,EC50,Ki&limit={limit}"
    try:
        r = requests.get(url, timeout=30)
        if not r.ok:
            print(f"Failed to fetch {label}: HTTP {r.status_code}")
            return
        data = r.json()
        acts = data.get("activities", [])
        print(f"  Retrieved {len(acts)} activities for {label}")
        
        records = []
        for a in acts:
            smiles = a.get("canonical_smiles")
            val = a.get("standard_value")
            stype = a.get("standard_type")
            units = a.get("standard_units")
            mol_id = a.get("molecule_chembl_id")
            if smiles and val:
                records.append({
                    "molecule_chembl_id": mol_id,
                    "canonical_smiles": smiles,
                    "target_chembl_id": chembl_id,
                    "target_channel": label,
                    "standard_type": stype,
                    "standard_value_nM": float(val),
                    "standard_units": units,
                    "relation": a.get("standard_relation", "=")
                })
        
        df = pd.DataFrame(records)
        out_file = TARGET_DIR / f"chembl_{label.lower()}_bioactivities.csv"
        df.to_csv(out_file, index=False)
        print(f"  Saved {out_file.name} ({len(df)} compounds with SMILES)")
    except Exception as e:
        print(f"Error fetching {label}: {e}")

def main():
    for label, tid in TARGETS.items():
        fetch_target_bioactivities(label, tid, limit=200)

if __name__ == "__main__":
    main()
