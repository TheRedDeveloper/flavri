"""
parse_singles.py
Exhaustive ingestion and harmonization for all 14 datasets into unified single tastants (FLAVRI_TASTE_SINGLES).
Includes biophysically grounded negative supervision for mouthfeel and chemesthetic modalities.
"""

import os
import re
import math
import pypdf
import pyreadr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils import (
    CHANNELS,
    canonicalize_smiles,
    parse_peptide_sequence,
    threshold_to_potency,
    affinity_to_potency,
)

RAW_ROOT = Path(__file__).resolve().parent.parent / "exploration" / "raw"


def load_tpdb_derivatives_map() -> Dict[str, str]:
    """Load derivative peptide name -> SMILES mapping from TastePeptidesDB."""
    path = RAW_ROOT / "tastepeptides_db" / "1-tastepeptides_derivative-2024-0621.csv"
    mapping = {}
    if path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            name = str(row.get("PepName", "")).strip()
            smiles = str(row.get("Smiles", "")).strip()
            if name and smiles and smiles != "nan":
                mapping[name] = smiles
    return mapping


def load_jecfa_catalog() -> Dict[str, Dict[str, Any]]:
    """Parse JECFA rds flavor catalog for entity cross-referencing."""
    path = RAW_ROOT / "jecfa_db" / "output" / "jecfa.rds"
    catalog = {}
    if path.exists():
        try:
            result = pyreadr.read_r(str(path))
            df = list(result.values())[0]
            for _, r in df.iterrows():
                cas = str(r.get("CAS.number", "")).strip()
                name = str(r.get("JECFA_name", "")).strip()
                fema = str(r.get("FEMA.number", "")).strip()
                if cas and cas != "nan":
                    catalog[cas] = {"name": name, "fema": fema}
            print(f"[JECFA] Loaded {len(catalog)} CAS cross-references from JECFA catalog.")
        except Exception as e:
            print(f"[WARN] Failed to load JECFA: {e}")
    return catalog


def parse_chemtastesdb() -> List[Dict[str, Any]]:
    """Dataset 9: ChemTastesDB v2.1 (4,078 compounds, 783 disconnected salts)."""
    path = RAW_ROOT / "chemtastesdb" / "ChemTastesDB_database.xlsx"
    records = []
    if not path.exists():
        print(f"[WARN] Not found: {path}")
        return records

    df = pd.read_excel(path)
    for _, row in df.iterrows():
        smi = canonicalize_smiles(row.get("canonical SMILES"))
        if not smi:
            continue
        
        cls = str(row.get("Class taste", "")).strip()
        taste_str = str(row.get("Taste", "")).lower()
        rec = {"smiles": smi, "source": "ChemTastesDB", "observations": {}}
        
        if "bitter" in cls.lower() or "bitter" in taste_str:
            rec["observations"]["bitter"] = 1.0
        if "sweet" in cls.lower() or "sweet" in taste_str:
            rec["observations"]["sweet"] = 1.0
        if "sour" in cls.lower() or "sour" in taste_str:
            rec["observations"]["sour"] = 1.0
        if "salt" in cls.lower() or "salty" in taste_str:
            rec["observations"]["salty"] = 1.0
        if "umami" in cls.lower() or "umami" in taste_str:
            rec["observations"]["umami"] = 1.0
            
        if cls == "Tastelessness" or "tasteless" in taste_str:
            # Confirmed tasteless across all 11 channels!
            for ch in CHANNELS:
                rec["observations"][ch] = 0.0

        if cls == "Non-sweetness" or "non-sweet" in taste_str:
            rec["observations"]["sweet"] = 0.0
        if cls == "Non-bitterness" or "non-bitter" in taste_str:
            rec["observations"]["bitter"] = 0.0

        if rec["observations"]:
            records.append(rec)

    print(f"[ChemTastesDB] Parsed {len(records)} records.")
    return records


def parse_bitterdb() -> List[Dict[str, Any]]:
    """Dataset 10: BitterDB 2024 Release (2,250 compounds + thresholds + EC50)."""
    path = RAW_ROOT / "bitterdb" / "BitterCompoundsPropA_2024.csv"
    records = []
    if not path.exists():
        print(f"[WARN] Not found: {path}")
        return records

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        smi = canonicalize_smiles(row.get("canonical_smiles"))
        if not smi:
            continue

        potency = 1.0
        thr = row.get("SensoryThr")
        try:
            thr_float = float(thr)
            if thr_float > 0:
                potency = threshold_to_potency(thr_float, unit="uM", pT_mid=-1.0, slope=0.8)
        except Exception:
            pass

        records.append({
            "smiles": smi,
            "source": "BitterDB_2024",
            "observations": {"bitter": potency, "sweet": 0.0}
        })

    print(f"[BitterDB] Parsed {len(records)} records.")
    return records


def parse_multifavor() -> List[Dict[str, Any]]:
    """Dataset 11: BitterGNN Multi-Flavor."""
    mf_dir = RAW_ROOT / "peptides_heyigacu" / "bittergnn" / "dataset" / "taste_dataset" / "multi_flavor"
    records = []
    if not mf_dir.exists():
        print(f"[WARN] Not found: {mf_dir}")
        return records

    class_to_channel = {
        "Sweet": "sweet",
        "Bitter": "bitter",
        "Sour": "sour",
        "Salt": "salty",
        "Umami": "umami",
        "Kokumi": "kokumi",
        "Astringent": "astringent",
    }

    for csv_file in mf_dir.glob("*.csv"):
        if csv_file.name == "sampled.csv":
            continue
        stem = csv_file.stem
        try:
            df = pd.read_csv(csv_file, sep="\t" if "\t" in open(csv_file, encoding="utf-8", errors="ignore").readline() else ",")
        except Exception:
            continue
        
        for _, row in df.iterrows():
            smi = canonicalize_smiles(row.get("Smiles") or row.get("smiles"))
            if not smi:
                continue

            obs = {}
            if stem == "Tasteless":
                # Confirmed tasteless across all 11 channels
                for ch in CHANNELS:
                    obs[ch] = 0.0
            elif stem in class_to_channel:
                channel = class_to_channel[stem]
                obs[channel] = 1.0

            if obs:
                records.append({
                    "smiles": smi,
                    "source": f"MultiFlavor_{stem}",
                    "observations": obs
                })

    print(f"[Multi-Flavor] Parsed {len(records)} records.")
    return records


def parse_tpdb(deriv_dict: Dict[str, str]) -> List[Dict[str, Any]]:
    """Dataset 6: TastePeptidesDB Export."""
    path = RAW_ROOT / "tastepeptides_db" / "tpdb_export.csv"
    records = []
    if not path.exists():
        print(f"[WARN] Not found: {path}")
        return records

    taste_map = {
        "umami": "umami",
        "bitter": "bitter",
        "sour": "sour",
        "sweet": "sweet",
        "salty": "salty",
        "salt": "salty",
        "kokumi": "kokumi",
        "astringent": "astringent",
    }

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        pep_name = str(row.get("PepName", "")).strip()
        smi = parse_peptide_sequence(pep_name, deriv_dict)
        if not smi:
            continue

        taste = str(row.get("Taste", "")).strip().lower()
        obs = {}

        potency = 1.0
        thr = row.get("Threshold (mmol/L)")
        try:
            thr_float = float(thr)
            if thr_float > 0:
                potency = threshold_to_potency(thr_float, unit="mM", pT_mid=0.0, slope=1.0)
        except Exception:
            pass

        if taste == "tasteless":
            for ch in CHANNELS:
                obs[ch] = 0.0
        elif taste in taste_map:
            obs[taste_map[taste]] = potency

        if obs:
            records.append({
                "smiles": smi,
                "source": "TPDB_export",
                "observations": obs
            })

    print(f"[TPDB] Parsed {len(records)} records.")
    return records


def parse_tpdb_derivatives() -> List[Dict[str, Any]]:
    """Dataset 7: TPDB Derivatives."""
    path = RAW_ROOT / "tastepeptides_db" / "1-tastepeptides_derivative-2024-0621.csv"
    records = []
    if not path.exists():
        return records

    taste_map = {
        "umami": "umami", "bitter": "bitter", "sour": "sour",
        "sweet": "sweet", "salty": "salty", "salt": "salty",
        "kokumi": "kokumi", "astringent": "astringent",
    }

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        smi = canonicalize_smiles(row.get("Smiles"))
        if not smi:
            continue

        taste = str(row.get("Intrinsic Taste", "")).strip().lower()
        thr = row.get("Intrinsic Threshold(mM)")
        potency = 1.0
        try:
            thr_float = float(thr)
            if thr_float > 0:
                potency = threshold_to_potency(thr_float, unit="mM", pT_mid=0.0, slope=1.0)
        except Exception:
            pass

        obs = {}
        if taste in taste_map:
            obs[taste_map[taste]] = potency

        if obs:
            records.append({
                "smiles": smi,
                "source": "TPDB_derivatives",
                "observations": obs
            })

    print(f"[TPDB Derivatives] Parsed {len(records)} records.")
    return records


def parse_umamimeta() -> List[Dict[str, Any]]:
    """Dataset 8A: UmamiMeta 2024."""
    path = RAW_ROOT / "peptides_heyigacu" / "umamimeta" / "code" / "dock" / "ummpep2024.csv"
    records = []
    if not path.exists():
        return records

    df = pd.read_csv(path, sep="\t" if "\t" in open(path).readline() else ",")
    for _, row in df.iterrows():
        seq = str(row.get("Sequence", "")).strip()
        smi = parse_peptide_sequence(seq)
        if not smi:
            continue
        try:
            umami_label = int(row.get("umami", 1))
            records.append({
                "smiles": smi,
                "source": "UmamiMeta_2024",
                "observations": {"umami": 1.0 if umami_label == 1 else 0.0}
            })
        except Exception:
            pass

    print(f"[UmamiMeta] Parsed {len(records)} records.")
    return records


def parse_virtuous_umami() -> List[Dict[str, Any]]:
    """Dataset 8B: VirtuousUmami (Train & Test)."""
    records = []
    for fname in ["Umami_training_dataset.csv", "Umami_testing_dataset.csv"]:
        path = RAW_ROOT / "virtuous_umami" / "data" / fname
        if not path.exists():
            continue

        df = pd.read_csv(path)
        for _, row in df.iterrows():
            smi = canonicalize_smiles(row.get("Std_SMILES") or row.get("Parent_SMILES"))
            if not smi:
                continue
            taste = str(row.get("TASTE", "")).strip().lower()
            records.append({
                "smiles": smi,
                "source": f"VirtuousUmami_{fname[:5]}",
                "observations": {"umami": 1.0 if taste == "umami" else 0.0}
            })

    print(f"[VirtuousUmami] Parsed {len(records)} records.")
    return records


def _read_fasta_sequences(fpath: Path) -> List[str]:
    """Helper to read sequences from FASTA or line-delimited text."""
    if not fpath.exists():
        return []
    seqs = []
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        curr = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr:
                    seqs.append("".join(curr))
                    curr = []
            else:
                curr.append(line)
        if curr:
            seqs.append("".join(curr))
    return seqs


def parse_iumami_benchmark() -> List[Dict[str, Any]]:
    """Dataset 8C: Authentic UMP442 iUmami Benchmark (302 non-umami, 140 umami)."""
    iu_dir = RAW_ROOT / "iumami_benchmark" / "iUmami"
    records = []
    if not iu_dir.exists():
        print(f"[WARN] Not found: {iu_dir}")
        return records

    neg_files = [("train-negative.txt", "iUmami_TR_neg"), ("test-negative.txt", "iUmami_TS_neg")]
    pos_files = [("train-positive.txt", "iUmami_TR_pos"), ("test-positive.txt", "iUmami_TS_pos")]

    for fname, src in neg_files:
        for seq in _read_fasta_sequences(iu_dir / fname):
            smi = parse_peptide_sequence(seq)
            if smi:
                records.append({
                    "smiles": smi,
                    "source": src,
                    "observations": {"umami": 0.0}
                })

    for fname, src in pos_files:
        for seq in _read_fasta_sequences(iu_dir / fname):
            smi = parse_peptide_sequence(seq)
            if smi:
                records.append({
                    "smiles": smi,
                    "source": src,
                    "observations": {"umami": 1.0}
                })

    print(f"[iUmami UMP442] Parsed {len(records)} authentic records.")
    return records


def parse_ibitter() -> List[Dict[str, Any]]:
    """Dataset 8D: BERT4Bitter / iBitter Benchmark (320 non-bitter, 320 bitter peptides)."""
    ib_dir = RAW_ROOT / "iumami_benchmark" / "iBitter"
    records = []
    if not ib_dir.exists():
        print(f"[WARN] Not found: {ib_dir}")
        return records

    neg_files = [("train-negative.fasta", "iBitter_TR_neg"), ("test-negative.fasta", "iBitter_TS_neg")]
    pos_files = [("train-positive.fasta", "iBitter_TR_pos"), ("test-positive.fasta", "iBitter_TS_pos")]

    for fname, src in neg_files:
        for seq in _read_fasta_sequences(ib_dir / fname):
            smi = parse_peptide_sequence(seq)
            if smi:
                records.append({
                    "smiles": smi,
                    "source": src,
                    "observations": {"bitter": 0.0}
                })

    for fname, src in pos_files:
        for seq in _read_fasta_sequences(ib_dir / fname):
            smi = parse_peptide_sequence(seq)
            if smi:
                records.append({
                    "smiles": smi,
                    "source": src,
                    "observations": {"bitter": 1.0}
                })

    print(f"[iBitter] Parsed {len(records)} records (320 non-bitter, 320 bitter).")
    return records


def parse_sensoproteomics_singles() -> List[Dict[str, Any]]:
    """Dataset 5: Sensoproteomics Pea Peptides (Table S8)."""
    path = RAW_ROOT / "synergy_sensoproteomics" / "jf4c02317_si_001.pdf"
    records = []
    if not path.exists():
        return records

    try:
        reader = pypdf.PdfReader(str(path))
        if len(reader.pages) >= 19:
            lines = reader.pages[18].extract_text().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 6 and parts[1].isdigit() and parts[2].startswith(('F', 'U')):
                    seq = parts[0]
                    umami_pred = parts[4].lower()
                    bitter_pred = parts[6].lower() if len(parts) > 6 else ""
                    smi = parse_peptide_sequence(seq)
                    if not smi:
                        continue
                    obs = {}
                    if "umami" in umami_pred:
                        obs["umami"] = 1.0
                    if "bitter" in bitter_pred and "non" not in bitter_pred:
                        obs["bitter"] = 1.0
                    elif "non" in bitter_pred:
                        obs["bitter"] = 0.0
                    if obs:
                        records.append({
                            "smiles": smi,
                            "source": "Sensoproteomics_TableS8",
                            "observations": obs
                        })
    except Exception as e:
        print(f"[WARN] Failed to parse Sensoproteomics PDF: {e}")

    print(f"[Sensoproteomics] Parsed {len(records)} peptide records.")
    return records


def parse_chembl_receptors() -> List[Dict[str, Any]]:
    """Dataset 13A: ChEMBL sensory channel bioactivities (TRPM8, TRPV1, TRPA1, CaSR)."""
    chembl_dir = RAW_ROOT / "chembl_receptors"
    records = []
    if not chembl_dir.exists():
        return records

    file_to_channel = {
        "chembl_trpm8_cooling_bioactivities.csv": "cooling",
        "chembl_trpv1_heat_bioactivities.csv": "heat",
        "chembl_trpa1_pungency_bioactivities.csv": "pungency",
        "chembl_casr_kokumi_bioactivities.csv": "kokumi",
    }

    for f_name, channel in file_to_channel.items():
        p = chembl_dir / f_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            smi = canonicalize_smiles(row.get("canonical_smiles"))
            if not smi:
                continue

            std_type = str(row.get("standard_type", "")).upper()
            val = row.get("standard_value_nM")
            relation = str(row.get("relation", "=")).strip()

            try:
                val_float = float(val)
                if std_type == "EC50" and relation in ("=", "<="):
                    potency = affinity_to_potency(val_float, unit="nM", pEC_min=4.0, pEC_max=9.0)
                elif relation == ">" or (std_type == "IC50" and val_float > 50000):
                    potency = 0.0
                else:
                    potency = affinity_to_potency(val_float, unit="nM", pEC_min=4.0, pEC_max=9.0)
                
                records.append({
                    "smiles": smi,
                    "source": f"ChEMBL_{channel}",
                    "observations": {channel: potency}
                })
            except Exception:
                pass

    print(f"[ChEMBL Receptors] Parsed {len(records)} records.")
    return records


def parse_specialized_benchmarks() -> List[Dict[str, Any]]:
    """Dataset 13B & 12: Alkylamides, Acid Speciation, Synthetic Coolants, Senomyx, SweetenersDB."""
    records = []

    # 1. Tingling Alkylamides Suite (14 sanshools / spilanthol)
    alk_p = RAW_ROOT / "tingling_alkylamides" / "leibniz_tingling_alkylamides_suite.csv"
    if alk_p.exists():
        df = pd.read_csv(alk_p)
        for _, row in df.iterrows():
            smi = canonicalize_smiles(row.get("canonical_smiles") or row.get("smiles"))
            if not smi:
                continue
            thr = row.get("tingling_detection_threshold_umol_L")
            potency = threshold_to_potency(thr, unit="uM", pT_mid=-1.0, slope=0.8)
            records.append({
                "smiles": smi,
                "source": "Leibniz_Alkylamides",
                "observations": {"tingling": potency}
            })

    # 2. Acid Speciation Benchmark (18 organic & mineral acids)
    acid_p = RAW_ROOT / "acid_sourness_speciation" / "acid_sourness_speciation_benchmark.csv"
    if acid_p.exists():
        df = pd.read_csv(acid_p)
        for _, row in df.iterrows():
            smi = canonicalize_smiles(row.get("smiles"))
            if not smi:
                continue
            thr = row.get("human_dt_mM")
            potency = threshold_to_potency(thr, unit="mM", pT_mid=0.0, slope=1.0)
            records.append({
                "smiles": smi,
                "source": "Acid_Speciation",
                "observations": {"sour": potency}
            })

    # 3. Synthetic Cooling Agents Compendium (WS coolants, icilin)
    cool_p = RAW_ROOT / "synthetic_cooling_compendium" / "synthetic_cooling_agents_compendium.csv"
    if cool_p.exists():
        df = pd.read_csv(cool_p)
        for _, row in df.iterrows():
            smi = canonicalize_smiles(row.get("smiles"))
            if not smi:
                continue
            rel = row.get("potency_relative_to_menthol")
            try:
                rel_float = float(rel)
                potency = max(0.5, min(1.0, 0.5 + 0.2 * math.log10(max(0.1, rel_float) + 1.0)))
            except Exception:
                potency = 0.8
            records.append({
                "smiles": smi,
                "source": "Synthetic_Coolants",
                "observations": {"cooling": potency}
            })

    # 4. Senomyx Synthetic Amides Library
    sen_p = RAW_ROOT / "senomyx_amides" / "senomyx_synthetic_amides_library.csv"
    if sen_p.exists():
        df = pd.read_csv(sen_p)
        for _, row in df.iterrows():
            smi = canonicalize_smiles(row.get("smiles"))
            if not smi:
                continue
            mod = str(row.get("modality", "")).lower()
            ec50 = row.get("ec50_uM")
            potency = affinity_to_potency(ec50, unit="uM", pEC_min=4.0, pEC_max=9.0)
            obs = {}
            if "umami" in mod:
                obs["umami"] = potency
            if "sweet" in mod:
                obs["sweet"] = potency
            if obs:
                records.append({
                    "smiles": smi,
                    "source": "Senomyx_Amides",
                    "observations": obs
                })

    # 5. SweetenersDB
    sweet_p = RAW_ROOT / "sweeteners" / "sweetenersdb_scraped.csv"
    if sweet_p.exists():
        df = pd.read_csv(sweet_p)
        sucrose_smi = canonicalize_smiles("C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O")
        for _, row in df.iterrows():
            name = str(row.get("name", "")).strip()
            log_s = row.get("log_s")
            try:
                potency = 1.0 / (1.0 + math.exp(-float(log_s) / 1.5))
            except Exception:
                potency = 0.8

            smi = canonicalize_smiles(name)
            if not smi and "sucrose" in name.lower() and sucrose_smi:
                smi = sucrose_smi

            if smi:
                records.append({
                    "smiles": smi,
                    "source": "SweetenersDB",
                    "observations": {"sweet": potency}
                })

    # 6. Physiological Grounded Negatives for Trigeminal & Mouthfeel
    # Expanded library of pure carbohydrates, polyols, mineral salts, clean amino acids,
    # approved high-potency sweeteners, and food acids known physiologically to elicit zero chemesthetic sensation
    reference_clean_tastants = [
        # Pure Monosaccharides & Alditols
        "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O",    # Sucrose
        "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",  # D-Glucose
        "C([C@@H]1[C@@H]([C@@H]([C@H](O1)O)O)O)O",         # D-Fructose
        "C([C@@H]1[C@@H]([C@@H]([C@@H]([C@H](O1)O)O)O)O)O", # D-Galactose
        "C([C@@H]1[C@H]([C@@H]([C@@H]([C@H](O1)O)O)O)O)O",   # D-Mannose
        "C1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O",             # D-Xylose
        "C1[C@H]([C@@H]([C@@H]([C@H](O1)O)O)O)O",           # L-Arabinose
        "C1[C@@H]([C@@H]([C@@H]([C@H](O1)O)O)O)O",           # D-Ribose
        "C([C@@H]1[C@H]([C@H]([C@H](O1)O)O)O)O",            # D-Tagatose
        "C([C@@H]1[C@@H]([C@H]([C@H](O1)O)O)O)O",           # D-Psicose (Allulose)
        "C[C@H]1[C@H]([C@@H]([C@@H]([C@H](O1)O)O)O)O",     # L-Rhamnose
        
        # Disaccharides & Oligosaccharides
        "C(C1C(C(C(C(O1)OC2C(C(C(OC2O)CO)O)O)O)O)O)O",      # Lactose
        "C(C1C(C(C(C(O1)OC2C(C(C(OC2O)CO)O)O)O)O)O)O",      # Maltose
        "C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O",    # Trehalose
        "C(C1C(C(C(C(O1)OCC2(C(C(C(O2)CO)O)O)CO)O)O)O)O",   # Isomaltulose (Palatinose)
        "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O",    # Lactulose
        
        # Polyols / Sugar Alcohols
        "C(C(C(CO)O)O)O",                                  # Erythritol
        "C(C(C(C(CO)O)O)O)O",                              # Xylitol
        "C(C(C(C(C(CO)O)O)O)O)O",                          # D-Sorbitol
        "C([C@@H]([C@@H]([C@H]([C@H](CO)O)O)O)O)O",        # D-Mannitol
        "C(C1C(C(C(C(O1)OCC(C(C(CO)O)O)O)O)O)O)O",        # Maltitol
        "C1(C(C(C(C(C1O)O)O)O)O)O",                        # myo-Inositol
        "OCC(O)CO",                                        # Glycerol
        "CC(O)CO",                                         # Propylene glycol
        "C([C@@H]([C@H](CO)O)O)O",                         # D-Threitol
        
        # Approved High-Potency Clean Sweeteners
        "COC(=O)[C@H](CC(=O)O)NC(=O)[C@H](Cc1ccccc1)N",     # Aspartame
        "CC(C)(C)CC(=O)N[C@H](CC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)OC", # Neotame
        "O=C1NS(=O)(=O)c2ccccc21",                         # Saccharin
        "O=S(=O)([O-])NC1CCCCC1.[Na+]",                    # Sodium cyclamate
        "CC1=CC(=O)N(C(=O)O1)[K+]",                        # Acesulfame-K
        
        # Mineral Salts & Non-Trigeminal Buffer Species
        "[Cl-].[Na+]",                                      # NaCl
        "[Cl-].[K+]",                                       # KCl
        "[Cl-].[Ca+2].[Cl-]",                               # CaCl2
        "[Cl-].[Mg+2].[Cl-]",                               # MgCl2
        "[Na+].[Na+].[O-]S(=O)(=O)[O-]",                    # Na2SO4
        "[K+].[K+].[O-]S(=O)(=O)[O-]",                      # K2SO4
        "[Na+].[O-]C(=O)C(O)C(O)C(O)C(O)CO",               # Sodium gluconate
        "[Na+].[O-]C(=O)C(C)O",                             # Sodium lactate
        "[Na+].[O-]C(=O)CC(O)(CC(=O)[O-])C(=O)[O-].[Na+].[Na+]", # Trisodium citrate
        "[K+].[O-]C(=O)C(C)O",                              # Potassium lactate
        "[K+].[O-]C(=O)CC(O)(CC(=O)[O-])C(=O)[O-].[K+].[K+]", # Tripotassium citrate
        "[Na+].[O-]C(=O)C",                                 # Sodium acetate
        "[K+].[O-]C(=O)C",                                  # Potassium acetate
        
        # Physiological Clean Amino Acids
        "NCC(=O)O",                                        # Glycine
        "CC(N)C(=O)O",                                     # L-Alanine
        "C(C(C(=O)O)N)O",                                  # L-Serine
        "CC(C(C(=O)O)N)O",                                 # L-Threonine
        "C1CC(NC1)C(=O)O",                                 # L-Proline
        "C(CC(=O)N)C(C(=O)O)N",                            # L-Glutamine
        "C(C(C(=O)O)N)C(=O)N",                             # L-Asparagine
        "NCCS(=O)(=O)O",                                   # Taurine
        "CN(CC(=O)O)C(=N)N",                               # Creatine
        "C[N+](C)(C)CC(=O)[O-]",                           # Betaine
        
        # Clean Food Acids (Non-trigeminal)
        "OC(=O)CC(O)(CC(=O)O)C(=O)O",                      # Citric acid
        "OC(=O)CC(O)C(=O)O",                               # L-Malic acid
        "OC(=O)C(O)C(O)C(=O)O",                            # L-Tartaric acid
        "CC(O)C(=O)O",                                     # L-Lactic acid
        "OC(=O)C=CC(=O)O",                                 # Fumaric acid
        "OC(=O)CCC(=O)O",                                  # Succinic acid
        "OCC(O)C(O)C(O)C(O)C(=O)O",                        # Gluconic acid
        "OCC(=O)O",                                        # Glycolic acid
        "OC(=O)CCCCC(=O)O",                                # Adipic acid
        "OC(=O)CCCC(=O)O",                                 # Glutaric acid
    ]
    for raw_ref in reference_clean_tastants:
        smi = canonicalize_smiles(raw_ref)
        if smi:
            records.append({
                "smiles": smi,
                "source": "Physiological_Orthogonal_Negatives",
                "observations": {
                    "astringent": 0.0,
                    "cooling": 0.0,
                    "heat": 0.0,
                    "pungency": 0.0,
                    "tingling": 0.0,
                }
            })

    print(f"[Specialized Benchmarks] Parsed {len(records)} records.")
    return records


def parse_bittergnn_assays() -> List[Dict[str, Any]]:
    """Dataset 14: BitterGNN Assays (bn_train/test and bs_train/test)."""
    records = []
    bitter_dir = RAW_ROOT / "peptides_heyigacu" / "bittergnn" / "dataset" / "taste_dataset"

    # 1. Bitter vs Non-Bitter (Label: 1 = Bitter, 0 = Non-Bitter)
    for fname in ["bn_train.csv", "bn_test.csv"]:
        p = bitter_dir / "bitter_nonbitter" / fname
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            for _, row in df.iterrows():
                smi = canonicalize_smiles(row.get("Smiles"))
                if not smi:
                    continue
                lbl = float(row.get("Label", 0))
                obs = {"bitter": 1.0 if lbl == 1.0 else 0.0}
                if lbl == 1.0:
                    obs["sweet"] = 0.0
                records.append({
                    "smiles": smi,
                    "source": f"BitterGNN_{fname[:2]}",
                    "observations": obs
                })

    # 2. Bitter vs Sweet (Label: 1 = Bitter, 0 = Sweet)
    for fname in ["bs_train.csv", "bs_test.csv"]:
        p = bitter_dir / "bitter_sweet" / fname
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            for _, row in df.iterrows():
                smi = canonicalize_smiles(row.get("Smiles"))
                if not smi:
                    continue
                lbl = float(row.get("Label", 0))
                # Label 1 is bitter (and not sweet), Label 0 is sweet (and not bitter)
                obs = {
                    "bitter": 1.0 if lbl == 1.0 else 0.0,
                    "sweet": 0.0 if lbl == 1.0 else 1.0,
                }
                records.append({
                    "smiles": smi,
                    "source": f"BitterGNN_{fname[:2]}",
                    "observations": obs
                })

    print(f"[BitterGNN Assays] Parsed {len(records)} records.")
    return records


def parse_n6512_bitter_screen() -> List[Dict[str, Any]]:
    """Dataset 15: N6512 Bitter vs Non-Bitter Public Screen (Label: 1 = Bitter, 0 = Non-Bitter)."""
    path = RAW_ROOT / "peptides_heyigacu" / "bittergnn" / "dataset" / "public_dataset" / "classification" / "N6512.csv"
    records = []
    if not path.exists():
        return records

    df = pd.read_csv(path, sep="\t")
    for _, row in df.iterrows():
        smi = canonicalize_smiles(row.get("Smiles"))
        if not smi:
            continue
        lbl = float(row.get("Label", 0))
        obs = {"bitter": 1.0 if lbl == 1.0 else 0.0}
        if lbl == 1.0:
            obs["sweet"] = 0.0
        records.append({
            "smiles": smi,
            "source": "N6512_BitterScreen",
            "observations": obs
        })

    print(f"[N6512 Screen] Parsed {len(records)} records.")
    return records


def parse_jecfa_fema_tasteless() -> List[Dict[str, Any]]:
    """Dataset 16: FEMA GRAS & JECFA Tasteless Matrix Carriers (Multimodal Class 0)."""
    path = RAW_ROOT / "jecfa_db" / "jecfa_fema_tasteless_matrix.csv"
    records = []
    if not path.exists():
        return records
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        smi = canonicalize_smiles(row.get("smiles"))
        if not smi:
            continue
        # Verified organoleptically neutral/tasteless across basic tastes + chemesthesis
        obs = {
            "sweet": 0.0,
            "bitter": 0.0,
            "sour": 0.0,
            "salty": 0.0,
            "umami": 0.0,
            "kokumi": 0.0,
            "astringent": 0.0,
            "cooling": 0.0,
            "heat": 0.0,
            "pungency": 0.0,
            "tingling": 0.0,
        }
        records.append({
            "smiles": smi,
            "source": "FEMA_JECFA_Tasteless_Matrix",
            "observations": obs,
        })
    print(f"[FEMA/JECFA Tasteless] Parsed {len(records)} records.")
    return records


def parse_pubchem_chemesthesis_inactives() -> List[Dict[str, Any]]:
    """Dataset 17: PubChem High-Throughput Screening Inactives for TRPA1, TRPV1, TRPM8, ENaC."""
    records = []
    sources = [
        (RAW_ROOT / "chemesthesis_trpa1" / "pubchem_trpa1_inactives.csv", "pungency", "PubChem_TRPA1_Inactive"),
        (RAW_ROOT / "chemesthesis_trpv1" / "pubchem_trpv1_inactives.csv", "heat", "PubChem_TRPV1_Inactive"),
        (RAW_ROOT / "cooling_trpm8" / "pubchem_trpm8_inactives.csv", "cooling", "PubChem_TRPM8_Inactive"),
        (RAW_ROOT / "cooling_trpm8" / "chembl_trpm8_inactives.csv", "cooling", "ChEMBL_TRPM8_Inactive"),
        (RAW_ROOT / "enac_salty" / "pubchem_enac_inactives.csv", "salty", "PubChem_ENaC_Inactive"),
    ]
    for path, channel, src_name in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        count = 0
        for _, row in df.iterrows():
            smi = canonicalize_smiles(row.get("smiles"))
            if not smi:
                continue
            records.append({
                "smiles": smi,
                "source": src_name,
                "observations": {channel: 0.0}
            })
            count += 1
        print(f"[{src_name}] Parsed {count} inactives for {channel}.")
    return records


def parse_chembl_receptor_inactives() -> List[Dict[str, Any]]:
    """Dataset 18: ChEMBL Verified Receptor Inactives (CaSR -> Kokumi, TASK-1 -> Tingling)."""
    records = []
    sources = [
        (RAW_ROOT / "chembl_receptors" / "chembl_casr_inactives.csv", "kokumi", "ChEMBL_CaSR_Inactive"),
        (RAW_ROOT / "chembl_receptors" / "chembl_k2p_inactives.csv", "tingling", "ChEMBL_K2P_Inactive"),
    ]
    for path, channel, src_name in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        count = 0
        for _, row in df.iterrows():
            smi = canonicalize_smiles(row.get("smiles"))
            if not smi:
                continue
            records.append({
                "smiles": smi,
                "source": src_name,
                "observations": {channel: 0.0}
            })
            count += 1
        print(f"[{src_name}] Parsed {count} inactives for {channel}.")
    return records


def parse_otop1_sour_inactives() -> List[Dict[str, Any]]:
    """Dataset 19: OTOP1 Proton Channel Sour Inactives (Tu et al. Science 2018)."""
    path = RAW_ROOT / "acid_sourness_speciation" / "otop1_sour_inactives.csv"
    records = []
    if not path.exists():
        return records
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        smi = canonicalize_smiles(row.get("smiles"))
        if not smi:
            continue
        records.append({
            "smiles": smi,
            "source": "OTOP1_Sour_Inactive",
            "observations": {"sour": 0.0}
        })
    print(f"[OTOP1 Sour Inactives] Parsed {len(records)} records.")
    return records


def parse_salivary_prp_inactives() -> List[Dict[str, Any]]:
    """Dataset 20: Salivary PRP Non-Precipitating Monomeric Phenolics (Charlton / Hofmann)."""
    path = RAW_ROOT / "astringent_prp" / "salivary_prp_inactives.csv"
    records = []
    if not path.exists():
        return records
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        smi = canonicalize_smiles(row.get("smiles"))
        if not smi:
            continue
        records.append({
            "smiles": smi,
            "source": "Salivary_PRP_Inactive",
            "observations": {"astringent": 0.0}
        })
    print(f"[Salivary PRP Inactives] Parsed {len(records)} records.")
    return records


def ingest_all_singles() -> pd.DataFrame:
    """Run all singles parsers, merge by canonical SMILES, compute channel-wise mean and mask."""
    deriv_dict = load_tpdb_derivatives_map()
    jecfa_cat = load_jecfa_catalog()

    all_records = []
    all_records.extend(parse_chemtastesdb())
    all_records.extend(parse_bitterdb())
    all_records.extend(parse_multifavor())
    all_records.extend(parse_tpdb(deriv_dict))
    all_records.extend(parse_tpdb_derivatives())
    all_records.extend(parse_umamimeta())
    all_records.extend(parse_virtuous_umami())
    all_records.extend(parse_iumami_benchmark())
    all_records.extend(parse_ibitter())
    all_records.extend(parse_sensoproteomics_singles())
    all_records.extend(parse_chembl_receptors())
    all_records.extend(parse_chembl_receptor_inactives())
    all_records.extend(parse_pubchem_chemesthesis_inactives())
    all_records.extend(parse_otop1_sour_inactives())
    all_records.extend(parse_salivary_prp_inactives())
    all_records.extend(parse_jecfa_fema_tasteless())
    all_records.extend(parse_specialized_benchmarks())
    all_records.extend(parse_bittergnn_assays())
    all_records.extend(parse_n6512_bitter_screen())

    print(f"\n[Aggregation] Total raw observations gathered: {len(all_records)}")

    mol_map = {}
    for r in all_records:
        smi = r["smiles"]
        if smi not in mol_map:
            mol_map[smi] = {c: [] for c in CHANNELS}
            mol_map[smi]["sources"] = set()
        mol_map[smi]["sources"].add(r["source"])
        for ch, val in r["observations"].items():
            if ch in mol_map[smi]:
                mol_map[smi][ch].append(val)

    rows = []
    for smi, data in mol_map.items():
        row = {"smiles": smi}
        for ch in CHANNELS:
            vals = data[ch]
            if len(vals) > 0:
                row[f"y_{ch}"] = round(float(np.mean(vals)), 4)
                row[f"mask_{ch}"] = 1
            else:
                row[f"y_{ch}"] = 0.0
                row[f"mask_{ch}"] = 0
        row["num_sources"] = len(data["sources"])
        row["sources"] = ";".join(sorted(data["sources"]))
        rows.append(row)

    df_out = pd.DataFrame(rows)
    print(f"[Done] Total unique canonical molecules produced: {len(df_out)}")
    return df_out


if __name__ == "__main__":
    df = ingest_all_singles()
    out_dir = Path(__file__).resolve().parent.parent
    out_path = out_dir / "flavri_taste_singles.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved FLAVRI_TASTE_SINGLES to {out_path}")
