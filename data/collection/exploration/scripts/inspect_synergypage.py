"""
inspect_synergypage.py
Fetch and inspect the SynergyPage JS chunk to locate exact synergy API endpoints.
"""

import re
import urllib3
import requests
from pathlib import Path

urllib3.disable_warnings()

url = "http://www.tastepeptides-meta.com/assets/SynergyPage-o07CvZgo.js"
r = requests.get(url, verify=False, timeout=15)
print(f"SynergyPage size: {len(r.content)} bytes")

# Extract all API URLs or endpoints
endpoints = set(re.findall(r'["\'](/[a-zA-Z0-9_/.-]+)["\']', r.text))
print("All endpoints in SynergyPage chunk:")
for ep in sorted(endpoints):
    print("  ", ep)

# Also check for export or search or table URLs
base_api = "http://www.tastepeptides-meta.com/api/v1"
for ep in endpoints:
    if "api" in ep:
        target = "http://www.tastepeptides-meta.com" + ep
    else:
        target = base_api + ep
    try:
        res = requests.get(target, verify=False, timeout=5)
        print(f"Tested {target} -> HTTP {res.status_code}")
    except Exception as e:
        pass
