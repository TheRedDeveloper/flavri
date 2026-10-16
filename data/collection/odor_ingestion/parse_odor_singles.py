"""
parse_odor_singles.py
Constructs the canonical flavri_odor_singles.csv master table with explicit per-channel mask columns.
Integrates:
  1. TGSC-LF (The Good Scents Company + Leffingwell union) -> full 137 masks = 1
  2. MMD (Merged Molecular Dataset / Sajan et al. 2026, 6,711 molecules)
     - Maps only clean equivalents and taxonomic sublabels (lilac->floral, menthol->mint, etc.)
     - Discards 16 phantom/controversial terms (moss, bread, chicken, etc.)
     - Marks unassayed channels (sweet, nutty, leathery, etc.) as mask = 0 (unassayed)
  3. OlfactionBase (Sharma et al. 2022, 1,124 confirmed odorless molecules as full 137 zero anchors) -> full 137 masks = 1, y = 0.0
Master Schema:
  smiles, y_alcoholic..y_woody (137 float cols), mask_alcoholic..mask_woody (137 int cols), num_sources, sources (Total: 277 columns)
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from rdkit import Chem

from odor_ingestion.constants import (
    RAW_GOODSCENTS,
    RAW_LEFFINGWELL,
    RAW_MMD,
    RAW_OLFACTIONBASE_ODORLESS,
    OUTPUT_SINGLES_CSV,
    CANONICAL_ODOR_DESCRIPTORS,
    SYNONYM_MAP,
    MMD_MORPHOLOGICAL_MAP,
    MMD_CLEAN_SUBLABELS_AND_SYNONYMS,
    MMD_DISCARDED_PHANTOMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("parse_odor_singles")


def is_multi_different_organic(s: str) -> bool:
    """
    Returns True if SMILES contains multiple different organic fragments
    (e.g., unresolved commercial isomer mixtures).
    Preserves single organics with metal/salt counterions (e.g. sodium benzoate,
    calcium pantothenate) and pure inorganic minerals.
    """
    if "." not in s:
        return False
    frags = s.split(".")
    carbon_frags = []
    for f in frags:
        m = Chem.MolFromSmiles(f)
        if m and any(a.GetAtomicNum() == 6 for a in m.GetAtoms()):
            carbon_frags.append(Chem.MolToSmiles(m))
    return len(set(carbon_frags)) > 1


def canonicalize_smiles(s: str) -> str:
    """Standardize SMILES using RDKit."""
    m = Chem.MolFromSmiles(s.strip())
    if m is None:
        raise ValueError(f"Could not parse SMILES: {s}")
    return Chem.MolToSmiles(m)


def parse_odor_singles() -> pd.DataFrame:
    logger.info("Initializing unified odor singles parser with rigorous per-channel masks...")
    records = {}
    dropped_multi_organic = 0

    def init_or_get_record(cs: str):
        if cs not in records:
            records[cs] = {
                "sources": set(),
                "potencies": {d: 0.0 for d in CANONICAL_ODOR_DESCRIPTORS},
                "masks": {d: 0 for d in CANONICAL_ODOR_DESCRIPTORS},
            }
        return records[cs]

    # 1. Ingest The Good Scents Company (TGSC) -> Full 137 coverage (all masks = 1)
    if RAW_GOODSCENTS.exists():
        logger.info(f"Ingesting The Good Scents Company from {RAW_GOODSCENTS}...")
        df_gs = pd.read_csv(RAW_GOODSCENTS)
        for _, r in df_gs.iterrows():
            smi = str(r["IsomericSMILES"]).strip() if pd.notna(r.get("IsomericSMILES")) else ""
            if not smi or is_multi_different_organic(smi):
                dropped_multi_organic += 1
                continue
            try:
                cs = canonicalize_smiles(smi)
            except Exception:
                continue

            rec = init_or_get_record(cs)
            rec["sources"].add("GoodScents")
            for d in CANONICAL_ODOR_DESCRIPTORS:
                rec["masks"][d] = 1
                if d in r and float(r[d]) > 0.0:
                    rec["potencies"][d] = max(rec["potencies"][d], float(r[d]))
            for syn, target in SYNONYM_MAP.items():
                if syn in r and float(r[syn]) > 0.0:
                    rec["potencies"][target] = max(rec["potencies"][target], float(r[syn]))
        logger.info(f"Unique molecules after TGSC: {len(records)}")
    else:
        logger.warning(f"File not found: {RAW_GOODSCENTS}")

    # 2. Ingest Leffingwell -> Full 137 coverage (all masks = 1)
    if RAW_LEFFINGWELL.exists():
        logger.info(f"Ingesting Leffingwell PMP from {RAW_LEFFINGWELL}...")
        df_lf = pd.read_csv(RAW_LEFFINGWELL)
        for _, r in df_lf.iterrows():
            smi = str(r["IsomericSMILES"]).strip() if pd.notna(r.get("IsomericSMILES")) else ""
            if not smi or is_multi_different_organic(smi):
                dropped_multi_organic += 1
                continue
            try:
                cs = canonicalize_smiles(smi)
            except Exception:
                continue

            rec = init_or_get_record(cs)
            rec["sources"].add("Leffingwell")
            for d in CANONICAL_ODOR_DESCRIPTORS:
                rec["masks"][d] = 1
                if d in r and float(r[d]) > 0.0:
                    rec["potencies"][d] = max(rec["potencies"][d], float(r[d]))
            for syn, target in SYNONYM_MAP.items():
                if syn in r and float(r[syn]) > 0.0:
                    rec["potencies"][target] = max(rec["potencies"][target], float(r[syn]))
        logger.info(f"Unique molecules after TGSC + Leffingwell (TGSC-LF): {len(records)}")
    else:
        logger.warning(f"File not found: {RAW_LEFFINGWELL}")

    # 3. Ingest MMD (Merged Molecular Dataset / Sajan et al. 2026)
    # Build complete MMD mapping dictionary (only exact matches, morphological, and clean sublabels)
    if RAW_MMD.exists():
        logger.info(f"Ingesting MMD (Hierarchies of Smell 2026) from {RAW_MMD}...")
        df_mmd = pd.read_csv(RAW_MMD)
        mmd_cols = [c for c in df_mmd.columns if c not in ["CID", "IsomericSMILES", "Descriptors"]]
        canon_set = set(CANONICAL_ODOR_DESCRIPTORS)

        mmd_to_canon = {}
        for c in mmd_cols:
            if c in canon_set:
                mmd_to_canon[c] = c
            elif c in MMD_MORPHOLOGICAL_MAP:
                mmd_to_canon[c] = MMD_MORPHOLOGICAL_MAP[c]
            elif c in MMD_CLEAN_SUBLABELS_AND_SYNONYMS:
                mmd_to_canon[c] = MMD_CLEAN_SUBLABELS_AND_SYNONYMS[c]
            elif c in MMD_DISCARDED_PHANTOMS:
                pass  # Discard phantom terms cleanly

        mmd_assayed_canon_channels = set(mmd_to_canon.values())
        logger.info(f"MMD evaluated canonical channels: {len(mmd_assayed_canon_channels)} / 137")
        logger.info(f"MMD unassayed channels (masked out to 0): {len(canon_set - mmd_assayed_canon_channels)}")

        for _, r in df_mmd.iterrows():
            smi = str(r["IsomericSMILES"]).strip() if pd.notna(r.get("IsomericSMILES")) else ""
            if not smi or is_multi_different_organic(smi):
                dropped_multi_organic += 1
                continue
            try:
                cs = canonicalize_smiles(smi)
            except Exception:
                continue

            rec = init_or_get_record(cs)
            rec["sources"].add("MMD")

            # For evaluated channels in MMD, mark mask = 1
            for tgt in mmd_assayed_canon_channels:
                rec["masks"][tgt] = 1

            # Populate potencies for mapped labels
            for col, tgt in mmd_to_canon.items():
                val = float(r[col])
                if val > 0.0:
                    rec["potencies"][tgt] = max(rec["potencies"][tgt], val)

        logger.info(f"Unique molecules after adding MMD: {len(records)}")
    else:
        logger.warning(f"File not found: {RAW_MMD}")

    # 4. Ingest OlfactionBase Confirmed Odorless Controls (Sharma et al. 2022)
    # Odorless across the entire sensory spectrum -> full 137 masks = 1, potencies = 0.0
    if RAW_OLFACTIONBASE_ODORLESS.exists():
        logger.info(f"Ingesting OlfactionBase confirmed odorless molecules from {RAW_OLFACTIONBASE_ODORLESS}...")
        df_od = pd.read_csv(RAW_OLFACTIONBASE_ODORLESS)
        added_pure_odorless = 0
        for _, r in df_od.iterrows():
            smi = str(r["SMILES"]).strip() if pd.notna(r.get("SMILES")) else ""
            if not smi or is_multi_different_organic(smi):
                dropped_multi_organic += 1
                continue
            try:
                cs = canonicalize_smiles(smi)
            except Exception:
                continue

            rec = init_or_get_record(cs)
            # If confirmed odorless, activate all 137 masks
            cur_sum = sum(rec["potencies"].values())
            if cur_sum == 0.0:
                rec["sources"].add("OlfactionBase")
                for d in CANONICAL_ODOR_DESCRIPTORS:
                    rec["masks"][d] = 1
                added_pure_odorless += 1

        logger.info(f"Added/confirmed {added_pure_odorless} odorless control molecules with full 137-channel zero anchors.")
        logger.info(f"Total unified molecules after OlfactionBase: {len(records)}")
    else:
        logger.warning(f"File not found: {RAW_OLFACTIONBASE_ODORLESS}")

    # Construct final master table
    formatted_rows = []
    for cs, data in sorted(records.items()):
        all_sources = sorted(list(data["sources"]))
        row = {
            "smiles": cs,
            "num_sources": len(all_sources),
            "sources": ";".join(all_sources),
        }
        for d in CANONICAL_ODOR_DESCRIPTORS:
            row[f"y_{d}"] = float(data["potencies"][d])
            row[f"mask_{d}"] = int(data["masks"][d])
        formatted_rows.append(row)

    df_singles = pd.DataFrame(formatted_rows)

    # Order columns strictly: smiles, 137 y_<desc>, 137 mask_<desc>, num_sources, sources (277 cols)
    ordered_cols = ["smiles"]
    for d in CANONICAL_ODOR_DESCRIPTORS:
        ordered_cols.append(f"y_{d}")
    for d in CANONICAL_ODOR_DESCRIPTORS:
        ordered_cols.append(f"mask_{d}")
    ordered_cols.extend(["num_sources", "sources"])

    df_singles = df_singles[ordered_cols]
    OUTPUT_SINGLES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_singles.to_csv(OUTPUT_SINGLES_CSV, index=False)

    y_cols = [f"y_{d}" for d in CANONICAL_ODOR_DESCRIPTORS]
    mask_cols = [f"mask_{d}" for d in CANONICAL_ODOR_DESCRIPTORS]
    total_zeros = (df_singles[y_cols].sum(axis=1) == 0.0).sum()
    full_mask_count = (df_singles[mask_cols].sum(axis=1) == 137).sum()

    logger.info(f"Dropped multi-organic mixtures: {dropped_multi_organic}")
    logger.info(
        f"Successfully generated {len(df_singles):,} total molecules "
        f"({len(df_singles) - total_zeros:,} active odorants, {total_zeros:,} confirmed odorless zero-vectors). "
        f"Full 137-channel coverage: {full_mask_count:,} molecules; "
        f"Partial coverage with unassayed channels masked to 0: {len(df_singles) - full_mask_count:,} molecules. "
        f"Total columns: {len(ordered_cols)} (137 y_ + 137 mask_ + provenance) -> {OUTPUT_SINGLES_CSV}"
    )
    return df_singles


if __name__ == "__main__":
    parse_odor_singles()
