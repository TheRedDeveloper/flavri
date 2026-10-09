"""
fetch_tpdb_synergy_files.py
Download the exact static synergy and derivative CSVs from TastePeptidesDB server.
"""

import urllib3
import requests
from pathlib import Path
import pandas as pd

urllib3.disable_warnings()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw" / "tastepeptides_db"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "1-tastepeptides_synergy-2024-0621.csv",
    "1-tastepeptides_derivative-2024-0621.csv",
    "0-tastepeptides_name.csv"
]

base_url = "http://www.tastepeptides-meta.com/statics/1-TPDB/"
headers = {"User-Agent": "Mozilla/5.0"}

for fname in FILES:
    target = RAW_DIR / fname
    url = base_url + fname
    print(f"Downloading {url} -> {fname}...")
    try:
        r = requests.get(url, verify=False, timeout=30, headers=headers)
        if r.status_code == 200:
            with open(target, "wb") as f:
                f.write(r.content)
            print(f"[SUCCESS] Saved {fname} ({target.stat().st_size} bytes)")
            try:
                df = pd.read_csv(target, encoding="utf-8-sig")
                print(f"  Shape: {df.shape}, Cols: {list(df.columns)}")
                print(df.head(2))
            except Exception as e:
                print("  Read error:", e)
        else:
            print(f"[FAILED] HTTP {r.status_code}")
    except Exception as e:
        print(f"[ERROR]: {e}")
