"""
find_tastepeptides_synergy.py
Locate and download the synergistic/enhancing peptide data from TastePeptides-Meta.
"""

import re
import urllib3
import requests
from pathlib import Path

urllib3.disable_warnings()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw" / "tastepeptides_db"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def find_synergy_endpoints():
    url = "http://www.tastepeptides-meta.com/assets/index-XBLQrd1n.js"
    r = requests.get(url, verify=False, timeout=15)
    text = r.text
    print(f"Loaded bundle ({len(text)} bytes)")

    # Find all /api or /v1 or /tpdb or other routes
    routes = set(re.findall(r'["\'](/api/v1/[a-zA-Z0-9_/.-]+|/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)["\']', text))
    print(f"Found {len(routes)} routes:")
    for rt in sorted(routes):
        print("  ", rt)

    # Let's test specific API endpoints against the backend
    base_api = "http://www.tastepeptides-meta.com/api/v1"
    headers = {"User-Agent": "Mozilla/5.0"}

    candidates = [
        "/synergy/list",
        "/synergy/export",
        "/enhancer/list",
        "/enhancer/export",
        "/derivatives/export",
        "/derivatives/list",
        "/bigbang/export",
        "/bigbang/list",
        "/tpdb/synergy",
        "/tpdb/derivatives",
        "/stats/overview"
    ]

    for c in candidates:
        api_url = base_api + c
        try:
            res = requests.get(api_url, verify=False, timeout=10, headers=headers)
            print(f"Endpoint {c} -> HTTP {res.status_code} ({len(res.content)} bytes)")
            if res.status_code == 200:
                print("   Preview:", res.text[:200])
                if "export" in c or "list" in c:
                    fname = c.strip("/").replace("/", "_") + ".json" if res.headers.get("Content-Type", "").startswith("application/json") else c.strip("/").replace("/", "_") + ".csv"
                    with open(RAW_DIR / fname, "wb") as f:
                        f.write(res.content)
                    print(f"   Saved {fname}!")
        except Exception as e:
            print(f"Endpoint {c} Error: {e}")

if __name__ == "__main__":
    find_synergy_endpoints()
