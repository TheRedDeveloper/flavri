"""
build_odor_dataset.py
Master orchestrator script to build the complete FLAVRI_ODOR dataset.
"""

import logging
import time
from odor_ingestion.parse_odor_singles import parse_odor_singles
from odor_ingestion.parse_odor_pairs import parse_odor_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_odor_dataset")


def main():
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("   Starting FLAVRI_ODOR Dataset Construction Pipeline            ")
    logger.info("=================================================================")

    logger.info("[Step 1/2] Ingesting Single Odorants (GS-LF)...")
    df_singles = parse_odor_singles()
    logger.info(f"Step 1 Complete: {len(df_singles)} single odorants generated.")

    logger.info("[Step 2/2] Ingesting Binary Mixture Pairs (Blended Pairs)...")
    df_pairs = parse_odor_pairs()
    logger.info(f"Step 2 Complete: {len(df_pairs)} mixture pairs generated.")

    elapsed = time.time() - start_time
    logger.info("=================================================================")
    logger.info(f"   FLAVRI_ODOR Construction Successfully Completed in {elapsed:.1f}s ")
    logger.info(f"   Singles: {len(df_singles):,} molecules | Pairs: {len(df_pairs):,} mixtures")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
