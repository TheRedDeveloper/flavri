"""
fetch_tastepeptides_api.py
Directly saves the complete 797 KB TPDB export from the official backend.
"""

import urllib3
import requests
from pathlib import Path
import pandas as pd

urllib3.disable_warnings()

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = BASE_DIR / "raw" / "tastepeptides_db"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

url = "http://www.tastepeptides-meta.com/api/v1/tpdb/export"
headers = {"User-Agent": "Mozilla/5.0"}

print(f"Downloading {url}...")
r = requests.get(url, verify=False, timeout=30, headers=headers)
if r.status_code == 200:
    out_file = TARGET_DIR / "tpdb_export.csv"
    with open(out_file, "wb") as f:
        f.write(r.content)
    print(f"Successfully saved {out_file.name} ({out_file.stat().st_size} bytes)")
    
    # Check structure
    try:
        df = pd.read_csv(out_file, encoding="utf-8-sig")
        print(f"Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Columns:", list(df.columns))
        print("Sample taste distribution:")
        for c in df.columns:
            if "taste" in c.lower() or "type" in c.lower() or "label" in c.lower():
                print(f"Col '{c}':\n", df[c].value_counts().head(10))
    except Exception as e:
        print("Read error:", e)
else:
    print(f"Failed with HTTP {r.status_code}")
