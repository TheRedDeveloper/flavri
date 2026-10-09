"""
fetch_sweetenersdb.py
Scrape and parse SweetenersDB table from Seb Fiorucci's portal.
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = BASE_DIR / "raw" / "sweeteners"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

def parse_sweetenersdb():
    url = "http://sebfiorucci.free.fr/SweetenersDB/sweetenersDB.html"
    print(f"Fetching {url}...")
    r = requests.get(url, timeout=15)
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        print("No table found!")
        return

    records = []
    for row in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) >= 4:
            records.append({
                "id": cols[0],
                "name": cols[1],
                "log_s": cols[3]
            })
    
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} sweetener records.")
    print(df.head(5))
    df.to_csv(TARGET_DIR / "sweetenersdb_scraped.csv", index=False)
    print("Saved to sweetenersdb_scraped.csv")

if __name__ == "__main__":
    parse_sweetenersdb()
