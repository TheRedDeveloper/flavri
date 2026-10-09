"""
fetch_bitterdb.py
Fetch all BitterDB 2024 tables directly from official portal.
"""

import urllib3
import requests
from pathlib import Path
import pandas as pd

urllib3.disable_warnings()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw" / "bitterdb"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "BitterCompoundsPropA_2024.csv",
    "compoundsnamesA_2024.csv",
    "ligandReceptorsA_2024.csv",
    "compRefA_2024.csv",
    "cbitterresourceA_2024.csv",
    "dbreferencesA_2024.csv",
    "drugbankcidA_2024.csv",
    "IUPHAR_cidA_2024.csv",
    "mutationData_2024.csv",
    "ReceptorSearchA_2024.csv",
    "snpData_2024.csv",
]

def download_bitterdb_files():
    base_url = "https://bitterdb.agri.huji.ac.il/downloads/2024/?file="
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for fname in FILES:
        target = RAW_DIR / fname
        if target.exists() and target.stat().st_size > 0:
            print(f"[EXISTS] {fname}")
            continue
        url = base_url + fname
        print(f"[DOWNLOADING] {url} -> {fname}...")
        try:
            r = requests.get(url, verify=False, timeout=30, headers=headers)
            if r.status_code == 200:
                with open(target, "wb") as f:
                    f.write(r.content)
                print(f"[SUCCESS] Downloaded {fname} ({target.stat().st_size} bytes)")
            else:
                print(f"[FAILED] {fname} HTTP {r.status_code}")
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

if __name__ == "__main__":
    download_bitterdb_files()
