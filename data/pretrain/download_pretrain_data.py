"""
data/pretrain/download_pretrain_data.py
Downloads and prepares 500,000 clean, diverse, unlabeled molecules
for FLAVRI Stage 0 self-supervised chemical pre-training.
Source: MOSES Molecular Benchmark Dataset (1.93M clean small molecules).
"""

from pathlib import Path
import sys
import urllib.request
import pandas as pd
from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRETRAIN_DIR = REPO_ROOT / "data" / "pretrain"
OUT_CSV = PRETRAIN_DIR / "unlabeled_500k.csv"

MOSES_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"


def main():
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.exists() and OUT_CSV.stat().st_size > 10_000_000:
        print(f"Pre-training dataset already exists at: {OUT_CSV}")
        print(f"Size: {OUT_CSV.stat().st_size / (1024*1024):.1f} MB")
        return

    print("=" * 70)
    print("DOWNLOADING STAGE 0 CHEMICAL PRE-TRAINING DATASET")
    print(f"Target file: {OUT_CSV}")
    print("=" * 70)

    raw_download_path = PRETRAIN_DIR / "moses_dataset_raw.csv"
    
    if not raw_download_path.exists():
        print(f"Downloading MOSES dataset from:\n  {MOSES_URL}")
        req = urllib.request.Request(MOSES_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(raw_download_path, "wb") as f_out:
            total_size = int(resp.headers.get("content-length", 84482588))
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=40),
                DownloadColumn(),
                TransferSpeedColumn(),
            ) as progress:
                task = progress.add_task("Downloading MOSES dataset", total=total_size)
                chunk_size = 1024 * 64
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    progress.update(task, advance=len(chunk))
        print("Download complete.")
    else:
        print(f"Found cached raw download at: {raw_download_path}")

    print("\nReading and sampling 500,000 clean SMILES...")
    df = pd.read_csv(raw_download_path, usecols=["SMILES"])
    df = df.rename(columns={"SMILES": "smiles"})
    df = df.dropna().drop_duplicates(subset=["smiles"])

    print(f"Total unique molecules available: {len(df):,}")
    target_count = min(500_000, len(df))
    df_sample = df.sample(n=target_count, random_state=42).reset_index(drop=True)

    df_sample.to_csv(OUT_CSV, index=False)
    print(f"Successfully saved {len(df_sample):,} molecules to:\n  {OUT_CSV}")
    print(f"File size: {OUT_CSV.stat().st_size / (1024*1024):.1f} MB")

    # Clean up raw download to save disk space
    if raw_download_path.exists():
        try:
            raw_download_path.unlink()
            print("Cleaned up temporary raw file.")
        except Exception:
            pass

    print("=" * 70)


if __name__ == "__main__":
    main()
