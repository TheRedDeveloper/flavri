"""
parse_odor_singles.py
Constructs the canonical flavri_odor_singles.csv master table without redundant mask columns.
"""

import sys
import logging
import pandas as pd
from rdkit import Chem

from odor_ingestion.constants import (
    RAW_GSLF_MERGED,
    RAW_GOODSCENTS,
    RAW_LEFFINGWELL,
    OUTPUT_SINGLES_CSV,
    CANONICAL_ODOR_DESCRIPTORS,
    SYNONYM_MAP,
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


def canonicalize_nostereo(s: str) -> str:
    """Standardize SMILES without stereochemistry for provenance lookup."""
    m = Chem.MolFromSmiles(s.strip())
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def parse_odor_singles() -> pd.DataFrame:
    logger.info(f"Loading raw GS-LF merged dataset: {RAW_GSLF_MERGED}")
    df_raw = pd.read_csv(RAW_GSLF_MERGED)
    logger.info(f"Raw rows: {len(df_raw)}")

    logger.info(f"Loading GoodScents and Leffingwell for provenance mapping...")
    df_gs = pd.read_csv(RAW_GOODSCENTS)
    df_lf = pd.read_csv(RAW_LEFFINGWELL)
    gs_set = set(canonicalize_nostereo(s) for s in df_gs["IsomericSMILES"] if canonicalize_nostereo(s))
    lf_set = set(canonicalize_nostereo(s) for s in df_lf["IsomericSMILES"] if canonicalize_nostereo(s))

    records = []
    dropped_multi_organic = 0

    for idx, row in df_raw.iterrows():
        raw_smiles = row["nonStereoSMILES"]
        if is_multi_different_organic(raw_smiles):
            dropped_multi_organic += 1
            continue

        can_smi = canonicalize_smiles(raw_smiles)
        ns_smi = canonicalize_nostereo(raw_smiles)

        # Provenance
        in_gs = ns_smi in gs_set
        in_lf = ns_smi in lf_set
        if in_gs and in_lf:
            sources = "GoodScents;Leffingwell"
            num_sources = 2
        elif in_gs:
            sources = "GoodScents"
            num_sources = 1
        elif in_lf:
            sources = "Leffingwell"
            num_sources = 1
        else:
            sources = "GoodScents"
            num_sources = 1

        record = {
            "smiles": can_smi,
            "num_sources": num_sources,
            "sources": sources,
        }

        # Build potency dictionary
        potencies = {}
        for desc in CANONICAL_ODOR_DESCRIPTORS:
            val = float(row.get(desc, 0.0))
            potencies[desc] = val

        # Apply synonym mappings:
        for syn, target in SYNONYM_MAP.items():
            if syn in row and float(row[syn]) > 0.0:
                potencies[target] = max(potencies.get(target, 0.0), float(row[syn]))

        # Format columns: y_<desc> (float) - no redundant mask columns
        for desc in CANONICAL_ODOR_DESCRIPTORS:
            record[f"y_{desc}"] = potencies[desc]

        records.append(record)

    df_singles = pd.DataFrame(records)
    logger.info(f"Dropped multi-organic mixtures: {dropped_multi_organic}")
    logger.info(f"Total curated singles before deduplication: {len(df_singles)}")

    # Deduplicate canonical SMILES if needed
    if df_singles["smiles"].duplicated().any():
        logger.info("Aggregating duplicate canonical SMILES...")
        grouped = []
        for smi, grp in df_singles.groupby("smiles"):
            all_sources = sorted(list(set(";".join(grp["sources"]).split(";"))))
            rec = {
                "smiles": smi,
                "num_sources": len(all_sources),
                "sources": ";".join(all_sources),
            }
            for d in CANONICAL_ODOR_DESCRIPTORS:
                rec[f"y_{d}"] = grp[f"y_{d}"].max()
            grouped.append(rec)
        df_singles = pd.DataFrame(grouped)

    # Order columns strictly: smiles, y_desc1, ..., y_desc137, num_sources, sources
    ordered_cols = ["smiles"]
    for d in CANONICAL_ODOR_DESCRIPTORS:
        ordered_cols.append(f"y_{d}")
    ordered_cols.extend(["num_sources", "sources"])

    df_singles = df_singles[ordered_cols]
    OUTPUT_SINGLES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_singles.to_csv(OUTPUT_SINGLES_CSV, index=False)
    logger.info(f"Successfully saved {len(df_singles)} single odorants (140 columns) to {OUTPUT_SINGLES_CSV}")
    return df_singles


if __name__ == "__main__":
    parse_odor_singles()
