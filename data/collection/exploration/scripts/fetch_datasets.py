"""
fetch_datasets.py
Systematic downloader for multimodal taste, chemesthesis, and mixture datasets.
"""

import os
import subprocess
import urllib.request
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def clone_repo(repo_url: str, target_subpath: str):
    target = RAW_DIR / target_subpath
    if target.exists():
        print(f"[EXISTS] {target_subpath} already cloned.")
        return
    print(f"[CLONING] {repo_url} -> {target_subpath}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target)],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[SUCCESS] Cloned {target_subpath}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to clone {repo_url}: {e.stderr}")

def download_file(url: str, target_subpath: str):
    target = RAW_DIR / target_subpath
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[EXISTS] {target_subpath} already downloaded.")
        return
    print(f"[DOWNLOADING] {url} -> {target_subpath}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp, open(target, "wb") as f:
            f.write(resp.read())
        print(f"[SUCCESS] Downloaded {target_subpath} ({target.stat().st_size} bytes)")
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")

def main():
    print("=== Starting Dataset Fetch ===")

    # 1. ChemTastesDB v2.1 from Zenodo
    download_file(
        "https://zenodo.org/api/records/15051366/files/ChemTastesDB_database.xlsx/content",
        "chemtastesdb/ChemTastesDB_database.xlsx"
    )
    download_file(
        "https://zenodo.org/api/records/15051366/files/ChemTastesDB_references.xlsx/content",
        "chemtastesdb/ChemTastesDB_references.xlsx"
    )

    # 2. TastePeptides-Meta
    clone_repo(
        "https://github.com/SynchronyML/Tastepeptides-meta-github.git",
        "tastepeptides_meta"
    )

    # 3. Heyigacu lab repositories (UmamiMeta, KokumiPD, acidmeta, astringentpd, BitterGNN)
    clone_repo("https://github.com/heyigacu/umamimeta.git", "peptides_heyigacu/umamimeta")
    clone_repo("https://github.com/heyigacu/KokumiPD.git", "peptides_heyigacu/kokumipd")
    clone_repo("https://github.com/heyigacu/acidmeta.git", "peptides_heyigacu/acidmeta")
    clone_repo("https://github.com/heyigacu/astringentpd.git", "peptides_heyigacu/astringentpd")
    clone_repo("https://github.com/heyigacu/BitterGNN.git", "peptides_heyigacu/bittergnn")

    # 4. VirtuousUmami (small molecules + peptides)
    clone_repo("https://github.com/lorenzopallante/VirtuousUmami.git", "virtuous_umami")

    # 5. iUmami Benchmark (UMP442)
    clone_repo("https://github.com/Shoombuatong2527/Benchmark-datasets.git", "iumami_benchmark")

    # 6. SaltyMeta (Salty peptides benchmark)
    clone_repo("https://github.com/chenwanchao/saltymeta-prediction-platform.git", "saltymeta")

    # 7. JECFA evaluation dataset
    clone_repo("https://github.com/UBESP-DCTV/jecfa.git", "jecfa_db")

    # 8. DeepSolNet / Sweet Proteins
    clone_repo("https://github.com/wangxinglong1990/DeepSolNet.git", "sweet_proteins_deepsolnet")

    print("=== Fetch Step Completed ===")

if __name__ == "__main__":
    main()
