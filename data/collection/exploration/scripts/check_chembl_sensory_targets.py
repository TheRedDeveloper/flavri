"""
check_chembl_sensory_targets.py
Find target ChEMBL IDs and activity counts for human TRP, TAS1R, and CASR receptors.
"""

import requests
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = BASE_DIR / "raw" / "chembl_receptors"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

genes = ["TRPM8", "TRPV1", "TRPA1", "TAS1R2", "TAS1R3", "CASR"]
results = {}

for gene in genes:
    url = f"https://www.ebi.ac.uk/chembl/api/data/target.json?target_synonym__icontains={gene}&organism=Homo%20sapiens"
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            targets = r.json().get("targets", [])
            if targets:
                t = targets[0]
                tid = t.get("target_chembl_id")
                pname = t.get("pref_name")
                
                # Query activity count
                act_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={tid}&limit=1"
                ract = requests.get(act_url, timeout=10)
                tot_acts = ract.json().get("page_meta", {}).get("total_count", 0) if ract.ok else "N/A"
                
                results[gene] = {
                    "chembl_id": tid,
                    "pref_name": pname,
                    "total_bioactivities": tot_acts
                }
                print(f"[{gene}] {tid} ({pname}): {tot_acts} bioactivities")
    except Exception as e:
        print(f"Error {gene}: {e}")

with open(TARGET_DIR / "chembl_sensory_targets.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved chembl_sensory_targets.json!")
