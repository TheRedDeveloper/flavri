"""
inspect_bundle_terms.py
Searches the JS bundle for occurrences of terms related to synergy, derivatives, etc.
"""

import re
import urllib3
import requests

urllib3.disable_warnings()

url = "http://www.tastepeptides-meta.com/assets/index-XBLQrd1n.js"
r = requests.get(url, verify=False, timeout=15)
text = r.text

for term in ["954", "synerg", "enhanc", "derivative", "bigbang"]:
    matches = [m.start() for m in re.finditer(term, text, re.IGNORECASE)]
    print(f"Term '{term}': {len(matches)} occurrences")
    for idx in matches[:2]:
        snippet = text[max(0, idx-80):min(len(text), idx+120)]
        print("  ...", snippet.replace("\n", " "), "...")
