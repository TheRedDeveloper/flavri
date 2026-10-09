"""
inspect_legacy_chunk.py
Fetch LegacyDatabaseOverview chunk and find the API endpoints used for synergy and derivatives.
"""

import re
import urllib3
import requests

urllib3.disable_warnings()

url = "http://www.tastepeptides-meta.com/assets/LegacyDatabaseOverview-BCkzA880.js"
r = requests.get(url, verify=False, timeout=15)
print(f"Legacy size: {len(r.content)} bytes")

# Find all endpoints
urls = set(re.findall(r'["\'](/api/v1/[^"\']+|/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)["\']', r.text))
print("All endpoints in Legacy chunk:")
for u in sorted(urls):
    print("  ", u)

# Search for mentions of synergy in this chunk
matches = [m.start() for m in re.finditer("synergy", r.text, re.IGNORECASE)]
print(f"Found {len(matches)} mentions of synergy:")
for idx in matches[:5]:
    snippet = r.text[max(0, idx-100):min(len(r.text), idx+150)]
    print("  ...", snippet.replace("\n", " "), "...")
