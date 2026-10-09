"""
fetch_ohsu_casr.py
Fetch the Ohsu et al. CaSR kokumi dataset from PLOS ONE API / web.
"""

import requests
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = BASE_DIR / "raw" / "kokumi_casr"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

def fetch_ohsu():
    doi = "10.1371/journal.pone.0034489"
    url = f"https://api.plos.org/search?q=doi:%22{doi}%22&fl=id,title,abstract,author_display"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    data = r.json()
    docs = data.get("response", {}).get("docs", [])
    if docs:
        doc = docs[0]
        print("Found article:", doc.get("title"))
        with open(TARGET_DIR / "ohsu_article_meta.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

    # Download XML full text from PLOS
    xml_url = f"https://journals.plos.org/plosone/article/file?id={doi}&type=manuscript"
    print(f"Downloading XML from {xml_url}...")
    rx = requests.get(xml_url, headers=headers, timeout=20)
    if rx.status_code == 200:
        with open(TARGET_DIR / "journal.pone.0034489.xml", "wb") as f:
            f.write(rx.content)
        print("Saved journal.pone.0034489.xml!")
        
        # Parse tables
        root = ET.fromstring(rx.content)
        tables = root.findall(".//table-wrap")
        print(f"Found {len(tables)} tables in manuscript XML:")
        for t in tables:
            lbl = t.find(".//label")
            cap = t.find(".//caption")
            print(" ", lbl.text if lbl is not None else "", ":", "".join(cap.itertext()) if cap is not None else "")
    else:
        print("Failed to download XML:", rx.status_code)

if __name__ == "__main__":
    fetch_ohsu()
