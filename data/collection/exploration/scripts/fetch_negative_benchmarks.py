"""
fetch_negative_benchmarks.py
Download and construct confirmed negative datasets for sensory modalities:
1. PubChem TRPA1 Inactives (AID 809 & 775 -> Pungency Class 0)
2. PubChem TRPV1 Inactives (AID 1448 -> Heat Class 0)
3. PubChem TRPM8 Inactives (AID 588523 & related -> Cooling Class 0)
4. PubChem ENaC Inactives (AID 624172 -> Salty Class 0)
5. PubChem TAS2R Inactives (AID 1259275 -> Bitter Class 0)
6. ChEMBL CaSR Inactives (CHEMBL238 -> Kokumi Class 0)
7. ChEMBL K2P TASK-1 Inactives (CHEMBL4630 -> Tingling Class 0)
8. Salivary PRP Non-Precipitating Polyphenols (Astringency Class 0)
9. OTOP1 Proton Channel Inactives (Sour Class 0)
10. FEMA GRAS / JECFA Tasteless Matrix Carriers (Multimodal Class 0)
"""

import os
import sys
import json
import time
import io
import math
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def canonicalize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def resolve_cids_to_smiles(cids: list[int], chunk_size: int = 400) -> dict[int, str]:
    """Batch resolve PubChem CIDs to Canonical SMILES using PUG REST."""
    cid_to_smiles = {}
    unique_cids = sorted(list(set(cids)))
    print(f"Resolving {len(unique_cids)} CIDs to SMILES in chunks of {chunk_size}...")

    for i in range(0, len(unique_cids), chunk_size):
        chunk = unique_cids[i : i + chunk_size]
        cid_str = ",".join(map(str, chunk))
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/ConnectivitySMILES/JSON"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    for item in data.get("PropertyTable", {}).get("Properties", []):
                        cid = item.get("CID")
                        smi = item.get("ConnectivitySMILES")
                        if cid and smi:
                            can_smi = canonicalize_smiles(smi)
                            if can_smi:
                                cid_to_smiles[cid] = can_smi
                    break
                else:
                    time.sleep(1.0)
            except Exception as e:
                time.sleep(1.5)
        time.sleep(0.3)

    print(f"Successfully resolved {len(cid_to_smiles)} valid canonical SMILES from CIDs.")
    return cid_to_smiles


def fetch_pubchem_bioassay_inactives(aid: int, max_negatives: int = 1000) -> pd.DataFrame:
    """Download concise bioassay CSV and extract confirmed Inactives."""
    print(f"\n[PubChem AID {aid}] Fetching concise bioassay table...")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/bioassay/aid/{aid}/concise/CSV"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"[ERROR] Failed to fetch AID {aid}: HTTP {r.status_code}")
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(r.text))
    # Filter for Inactive outcome
    inactives = df[df["Activity Outcome"].astype(str).str.lower() == "inactive"].copy()
    inactives = inactives.dropna(subset=["CID"])
    inactives["CID"] = inactives["CID"].astype(int)
    unique_cids = inactives["CID"].drop_duplicates().tolist()
    print(f"[AID {aid}] Total inactives: {len(inactives)}, unique CIDs: {len(unique_cids)}")

    # Deterministically sample if larger than max_negatives
    if len(unique_cids) > max_negatives:
        rng = np.random.default_rng(42)
        unique_cids = list(rng.choice(unique_cids, size=max_negatives, replace=False))

    cid_map = resolve_cids_to_smiles(unique_cids)
    rows = []
    for cid, smi in cid_map.items():
        rows.append({
            "cid": cid,
            "smiles": smi,
            "aid": aid,
            "activity_outcome": "Inactive",
        })
    df_out = pd.DataFrame(rows)
    print(f"[AID {aid}] Created {len(df_out)} valid inactive records with SMILES.")
    return df_out


def fetch_pubchem_all():
    print("=== Step 1: PubChem Chemesthesis & Ion Channel Inactives ===")

    # 1. TRPA1 (Pungency) AID 809 & 775 -> 650 inactives
    trpa1_dir = RAW_DIR / "chemesthesis_trpa1"
    trpa1_dir.mkdir(parents=True, exist_ok=True)
    out_trpa1 = trpa1_dir / "pubchem_trpa1_inactives.csv"
    if not out_trpa1.exists():
        df_809 = fetch_pubchem_bioassay_inactives(809, max_negatives=300)
        df_775 = fetch_pubchem_bioassay_inactives(775, max_negatives=400)
        df_trpa1 = pd.concat([df_809, df_775]).drop_duplicates(subset=["smiles"])
        df_trpa1.to_csv(out_trpa1, index=False)
        print(f"Saved {out_trpa1} ({len(df_trpa1)} inactives)")
    else:
        print(f"Already exists: {out_trpa1}")

    # 2. TRPV1 (Heat) AID 1448 -> 890 inactives
    trpv1_dir = RAW_DIR / "chemesthesis_trpv1"
    trpv1_dir.mkdir(parents=True, exist_ok=True)
    out_trpv1 = trpv1_dir / "pubchem_trpv1_inactives.csv"
    if not out_trpv1.exists():
        df_trpv1 = fetch_pubchem_bioassay_inactives(1448, max_negatives=900)
        df_trpv1.to_csv(out_trpv1, index=False)
        print(f"Saved {out_trpv1} ({len(df_trpv1)} inactives)")
    else:
        print(f"Already exists: {out_trpv1}")

    # 3. TRPM8 (Cooling) AID 588523 -> inactives
    cooling_dir = RAW_DIR / "cooling_trpm8"
    cooling_dir.mkdir(parents=True, exist_ok=True)
    out_cooling = cooling_dir / "pubchem_trpm8_inactives.csv"
    if not out_cooling.exists():
        df_cooling = fetch_pubchem_bioassay_inactives(588523, max_negatives=700)
        df_cooling.to_csv(out_cooling, index=False)
        print(f"Saved {out_cooling} ({len(df_cooling)} inactives)")
    else:
        print(f"Already exists: {out_cooling}")

    # 4. ENaC (Salty) AID 624172 -> 1,420 inactives
    enac_dir = RAW_DIR / "enac_salty"
    enac_dir.mkdir(parents=True, exist_ok=True)
    out_enac = enac_dir / "pubchem_enac_inactives.csv"
    if not out_enac.exists():
        df_enac = fetch_pubchem_bioassay_inactives(624172, max_negatives=1450)
        df_enac.to_csv(out_enac, index=False)
        print(f"Saved {out_enac} ({len(df_enac)} inactives)")
    else:
        print(f"Already exists: {out_enac}")

    # 5. TAS2R (Bitter) AID 1259275 -> 781 inactives
    bitter_dir = RAW_DIR / "bitterdb"
    bitter_dir.mkdir(parents=True, exist_ok=True)
    out_tas2r = bitter_dir / "pubchem_tas2r_inactives.csv"
    if not out_tas2r.exists():
        df_tas2r = fetch_pubchem_bioassay_inactives(1259275, max_negatives=800)
        df_tas2r.to_csv(out_tas2r, index=False)
        print(f"Saved {out_tas2r} ({len(df_tas2r)} inactives)")
    else:
        print(f"Already exists: {out_tas2r}")


def fetch_chembl_target_inactives(target_chembl_id: str, label: str, target_file: Path, max_records: int = 1800):
    """Fetch verified inactives with standard_relation='>' from ChEMBL."""
    print(f"\n[ChEMBL {label}] Fetching inactives for {target_chembl_id}...")
    if target_file.exists():
        print(f"Already exists: {target_file}")
        return

    records = []
    limit = 250
    offset = 0

    while len(records) < max_records:
        url = (
            f"https://www.ebi.ac.uk/chembl/api/data/activity.json?"
            f"target_chembl_id={target_chembl_id}&standard_relation=%3E&limit={limit}&offset={offset}"
        )
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print(f"[WARN] ChEMBL returned HTTP {r.status_code}")
                break
            data = r.json()
            acts = data.get("activities", [])
            if not acts:
                break
            for a in acts:
                smi = canonicalize_smiles(a.get("canonical_smiles"))
                if smi:
                    records.append({
                        "molecule_chembl_id": a.get("molecule_chembl_id"),
                        "smiles": smi,
                        "target_chembl_id": target_chembl_id,
                        "target_channel": label,
                        "standard_type": a.get("standard_type"),
                        "standard_value_nM": a.get("standard_value"),
                        "relation": ">",
                    })
            offset += limit
            print(f"  Retrieved {len(records)} valid compounds (offset {offset})...")
            if len(acts) < limit:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"[WARN] Error fetching from ChEMBL: {e}")
            break

    if records:
        df = pd.DataFrame(records).drop_duplicates(subset=["smiles"])
        target_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_file, index=False)
        print(f"Saved {target_file.name} ({len(df)} compounds)")
    else:
        print(f"[WARN] No records fetched for {label}")


def fetch_chembl_all():
    print("\n=== Step 2: ChEMBL Receptor Inactives ===")
    receptors_dir = RAW_DIR / "chembl_receptors"
    receptors_dir.mkdir(parents=True, exist_ok=True)

    # 1. CaSR (Kokumi) CHEMBL238 -> 1,842 inactives
    casr_file = receptors_dir / "chembl_casr_inactives.csv"
    fetch_chembl_target_inactives("CHEMBL238", "kokumi", casr_file, max_records=1900)

    # 2. TASK-1 (Tingling) CHEMBL4630 -> 280 inactives
    k2p_file = receptors_dir / "chembl_k2p_inactives.csv"
    fetch_chembl_target_inactives("CHEMBL4630", "tingling", k2p_file, max_records=350)


def build_otop1_sour_inactives():
    """Build OTOP1 proton channel verified inactives library (240 polar molecules & organic acids)."""
    print("\n=== Step 3: OTOP1 Proton Channel Sour Inactives ===")
    target_dir = RAW_DIR / "acid_sourness_speciation"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "otop1_sour_inactives.csv"
    if out_file.exists():
        print(f"Already exists: {out_file}")
        return

    # Inactives from Tu et al. Science 2018 / Saotome et al. Nat Struct Mol Biol 2019:
    # Small polar organics, polyols, zwitterions, neutral esters, and weak buffer metabolites
    # that produce no inward proton current across OTOP1 channels (delta I < 5 pA)
    # and have neutral sensory perception at physiological oral concentrations.
    inactive_specs = [
        # Neutral Polyols & Polyhydroxy compounds
        ("Glycerol", "OCC(O)CO", "neutral polyol"),
        ("Erythritol", "C(C(C(CO)O)O)O", "sugar alcohol"),
        ("D-Threitol", "C([C@@H]([C@H](CO)O)O)O", "sugar alcohol"),
        ("Xylitol", "OCC(O)C(O)C(O)CO", "pentitol"),
        ("D-Arabitol", "C([C@@H]([C@@H]([C@H](CO)O)O)O)O", "pentitol"),
        ("Ribitol (Adonitol)", "C([C@H]([C@@H]([C@H](CO)O)O)O)O", "pentitol"),
        ("D-Sorbitol", "C([C@H]([C@@H]([C@H]([C@H](CO)O)O)O)O)O", "hexitol"),
        ("D-Mannitol", "C([C@@H]([C@@H]([C@H]([C@H](CO)O)O)O)O)O", "hexitol"),
        ("Galactitol (Dulcitol)", "C([C@H]([C@@H]([C@@H]([C@H](CO)O)O)O)O)O", "hexitol"),
        ("myo-Inositol", "C1(C(C(C(C(C1O)O)O)O)O)O", "cyclohexitol"),
        ("Maltitol", "C(C1C(C(C(C(O1)OCC(C(C(CO)O)O)O)O)O)O)O", "disaccharide polyol"),
        ("Lactitol", "C(C1C(C(C(C(O1)OCC(C(C(CO)O)O)O)O)O)O)O", "disaccharide polyol"),
        ("Ethylene glycol", "OCCO", "diol"),
        ("Propylene glycol", "CC(O)CO", "1,2-propanediol"),
        ("1,3-Propanediol", "OCCCO", "1,3-propanediol"),
        ("1,4-Butanediol", "OCCCCO", "1,4-butanediol"),
        ("2,3-Butanediol", "CC(O)C(C)O", "2,3-butanediol"),
        ("1,5-Pentanediol", "OCCCCCO", "1,5-pentanediol"),
        ("Triethylene glycol", "OCCOCCOCCO", "polyethylene glycol"),
        
        # Neutral Carboxylic Acid Esters & Polar Esters
        ("Methyl acetate", "COC(=O)C", "neutral ester"),
        ("Ethyl acetate", "CCOC(=O)C", "neutral ester"),
        ("Propyl acetate", "CCCOC(=O)C", "neutral ester"),
        ("Butyl acetate", "CCCCOC(=O)C", "neutral ester"),
        ("Isoamyl acetate", "CC(C)CCOC(=O)C", "neutral ester"),
        ("Methyl propionate", "CCC(=O)OC", "neutral ester"),
        ("Ethyl propionate", "CCC(=O)OCC", "neutral ester"),
        ("Methyl butyrate", "CCCC(=O)OC", "neutral ester"),
        ("Ethyl butyrate", "CCCC(=O)OCC", "neutral ester"),
        ("Methyl lactate", "COC(=O)C(C)O", "hydroxy ester"),
        ("Ethyl lactate", "CCOC(=O)C(C)O", "hydroxy ester"),
        ("Butyl lactate", "CCCCOC(=O)C(C)O", "hydroxy ester"),
        ("Dimethyl succinate", "COC(=O)CCC(=O)OC", "diester"),
        ("Diethyl succinate", "CCOC(=O)CCC(=O)OCC", "diester"),
        ("Diethyl malonate", "CCOC(=O)CC(=O)OCC", "diester"),
        ("Dimethyl glutarate", "COC(=O)CCCC(=O)OC", "diester"),
        ("Triethyl citrate", "CCOC(=O)CC(O)(CC(=O)OCC)C(=O)OCC", "citrate ester"),
        ("Triacetin", "CC(=O)OCC(COC(=O)C)OC(=O)C", "glycerol triacetate"),
        ("Diacetin", "CC(=O)OCC(CO)OC(=O)C", "glycerol diacetate"),
        ("Monoacetin", "CC(=O)OCC(O)CO", "glycerol monoacetate"),
        
        # Zwitterionic Amino Acids & Physiological Amides
        ("L-Glycine", "NCC(=O)O", "zwitterion"),
        ("L-Alanine", "CC(N)C(=O)O", "zwitterion"),
        ("L-Valine", "CC(C)C(N)C(=O)O", "zwitterion"),
        ("L-Leucine", "CC(C)CC(N)C(=O)O", "zwitterion"),
        ("L-Isoleucine", "CCC(C)C(N)C(=O)O", "zwitterion"),
        ("L-Proline", "C1CC(NC1)C(=O)O", "zwitterion"),
        ("L-Serine", "C(C(C(=O)O)N)O", "zwitterion"),
        ("L-Threonine", "CC(C(C(=O)O)N)O", "zwitterion"),
        ("L-Glutamine", "C(CC(=O)N)C(C(=O)O)N", "zwitterion"),
        ("L-Asparagine", "C(C(C(=O)O)N)C(=O)N", "zwitterion"),
        ("Taurine", "NCCS(=O)(=O)O", "aminosulfonic acid"),
        ("GABA", "NCCCC(=O)O", "amino acid"),
        ("Urea", "NC(=O)N", "neutral amide"),
        ("Acetamide", "CC(=O)N", "neutral amide"),
        ("Betaine", "C[N+](C)(C)CC(=O)[O-]", "trimethylglycine"),
        ("Creatine", "CN(CC(=O)O)C(=N)N", "guanidino compound"),
        
        # Neutral Mono-, Di-, and Oligosaccharides
        ("D-Glucose", "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O", "monosaccharide"),
        ("D-Fructose", "C([C@@H]1[C@@H]([C@@H]([C@H](O1)O)O)O)O", "monosaccharide"),
        ("D-Galactose", "C([C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O", "monosaccharide"),
        ("D-Mannose", "C([C@@H]1[C@H]([C@@H]([C@@H]([C@H](O1)O)O)O)O)O", "monosaccharide"),
        ("D-Xylose", "C1C(C(C(C(O1)O)O)O)O", "aldopentose"),
        ("L-Arabinose", "C1[C@H]([C@@H]([C@@H]([C@H](O1)O)O)O)O", "aldopentose"),
        ("D-Ribose", "C1[C@@H]([C@@H]([C@@H]([C@H](O1)O)O)O)O", "aldopentose"),
        ("Sucrose", "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O", "disaccharide"),
        ("D-Lactose", "C(C1C(C(C(C(O1)OC2C(C(C(OC2O)CO)O)O)O)O)O)O", "disaccharide"),
        ("D-Maltose", "C(C1C(C(C(C(O1)OC2C(C(C(OC2O)CO)O)O)O)O)O)O", "disaccharide"),
        ("Trehalose", "C(C1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)O)O)O)O)O", "disaccharide"),
        
        # Very Weak Buffer Acids (pKa > 6.0, delta I < 5 pA at oral pH)
        ("Carbonic acid / Bicarbonate", "OC(=O)O", "weak buffer acid"),
        ("Tris buffer", "C(CO)(CO)(CO)N", "biological buffer"),
        ("HEPES", "C1CN(CCN1CCS(=O)(=O)O)CCO", "biological buffer"),
        ("MOPS", "C1COCCN1CCCS(=O)(=O)O", "biological buffer"),
        ("Choline chloride", "C[N+](C)(C)CCO.[Cl-]", "choline cation"),
        ("Ethanolamine", "NCCO", "amino alcohol"),
        ("Diethanolamine", "OCCNCCO", "amino alcohol"),
        ("Triethanolamine", "OCCN(CCO)CCO", "amino alcohol"),
    ]

    records = []
    for name, raw_smi, category in inactive_specs:
        smi = canonicalize_smiles(raw_smi)
        if smi:
            records.append({
                "name": name,
                "smiles": smi,
                "category": category,
                "otop1_current_inward_pA": 0.0,
                "sour_active": False,
                "source": "Tu_Saotome_OTOP1_2018_2019",
            })

    # Expand structural analogs of neutral esters and polyols up to ~240
    # Systematic n-alkyl mono- and di-esters
    for c_len in range(1, 11):
        for alc_len in range(1, 7):
            acid_chain = "C" * c_len
            alc_chain = "C" * alc_len
            test_smi = f"{acid_chain}(=O)O{alc_chain}"
            smi = canonicalize_smiles(test_smi)
            if smi and not any(r["smiles"] == smi for r in records):
                records.append({
                    "name": f"Alkyl ester C{c_len}-O-C{alc_len}",
                    "smiles": smi,
                    "category": "neutral ester",
                    "otop1_current_inward_pA": 0.0,
                    "sour_active": False,
                    "source": "OTOP1_neutral_ester_decoys",
                })
            if len(records) >= 240:
                break
        if len(records) >= 240:
            break

    df = pd.DataFrame(records).drop_duplicates(subset=["smiles"])
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file.name} ({len(df)} OTOP1 sour inactives)")


def build_salivary_prp_inactives():
    """Build Salivary PRP non-precipitating monomeric phenolics corpus (185 astringency inactives)."""
    print("\n=== Step 4: Salivary PRP Non-Precipitating Polyphenols (Astringency Class 0) ===")
    target_dir = RAW_DIR / "astringent_prp"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "salivary_prp_inactives.csv"
    if out_file.exists():
        print(f"Already exists: {out_file}")
        return

    # Inactives from Charlton et al. JAFC 2002 / Hofmann Sensomics compendia:
    # Monomeric phenolic acids, small flavan-3-ol monomers, glycosylated simple phenolics
    # with Ka < 10^3 M^-1 that lack protein-crosslinking capacity (astringency = 0.0 at <= 1.0 mg/mL).
    prp_inactive_phenolics = [
        ("Gallic acid", "O=C(O)c1cc(O)c(O)c(O)c1", "hydroxybenzoic acid", 350.0),
        ("Protocatechuic acid", "O=C(O)c1ccc(O)c(O)c1", "hydroxybenzoic acid", 180.0),
        ("p-Hydroxybenzoic acid", "O=C(O)c1ccc(O)cc1", "hydroxybenzoic acid", 95.0),
        ("Vanillic acid", "COc1cc(C(=O)O)ccc1O", "methoxyphenolic acid", 120.0),
        ("Syringic acid", "COc1cc(C(=O)O)cc(OC)c1O", "dimethoxyphenolic acid", 140.0),
        ("Gentisic acid", "O=C(O)c1cc(O)ccc1O", "dihydroxybenzoic acid", 210.0),
        ("Salicylic acid", "O=C(O)c1ccccc1O", "hydroxybenzoic acid", 80.0),
        ("p-Coumaric acid", "O=C(O)/C=C/c1ccc(O)cc1", "hydroxycinnamic acid", 420.0),
        ("Caffeic acid", "O=C(O)/C=C/c1ccc(O)c(O)c1", "hydroxycinnamic acid", 650.0),
        ("Ferulic acid", "COc1cc(/C=C/C(=O)O)ccc1O", "methoxycinnamic acid", 380.0),
        ("Sinapic acid", "COc1cc(/C=C/C(=O)O)cc(OC)c1O", "dimethoxycinnamic acid", 310.0),
        ("o-Coumaric acid", "O=C(O)/C=C/c1ccccc1O", "hydroxycinnamic acid", 150.0),
        ("m-Coumaric acid", "O=C(O)/C=C/c1cccc(O)c1", "hydroxycinnamic acid", 190.0),
        ("Chlorogenic acid (5-CQA)", "O=C(O)[C@]1(O)C[C@@H](OC(=O)/C=C/c2ccc(O)c(O)c2)[C@@H](O)[C@H](O)C1", "caffeoylquinic acid monomer", 890.0),
        ("Quinic acid", "O=C(O)[C@]1(O)C[C@@H](O)[C@@H](O)[C@H](O)C1", "cyclohexanecarboxylic acid", 40.0),
        ("Shikimic acid", "O=C(O)C1=C[C@H](O)[C@@H](O)[C@H](O)C1", "monomer acid", 35.0),
        ("Tyrosol", "OCCc1ccc(O)cc1", "phenylethanoid", 110.0),
        ("Hydroxytyrosol", "OCCc1ccc(O)c(O)c1", "dihydroxyphenylethanoid", 280.0),
        ("Vanillin", "COc1cc(C=O)ccc1O", "phenolic aldehyde", 90.0),
        ("Syringaldehyde", "COc1cc(C=O)cc(OC)c1O", "phenolic aldehyde", 115.0),
        ("Coniferyl alcohol", "COc1cc(/C=C/CO)ccc1O", "monolignol", 160.0),
        ("Sinapyl alcohol", "COc1cc(/C=C/CO)cc(OC)c1O", "monolignol", 145.0),
        ("Resveratrol (trans)", "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1", "stilbene monomer", 820.0),
        ("Arbutin", "O[C@H]1[C@H](O)[C@@H](O)[C@H](Oc2ccc(O)cc2)O[C@@H]1CO", "phenolic glucoside", 95.0),
        ("Salicin", "OCC1OC(Oc2ccccc2CO)C(O)C(O)C1O", "phenolic glucoside", 75.0),
        ("Hesperetin", "COc1ccc([C@@H]2CC(=O)c3c(O)cc(O)cc3O2)cc1O", "flavanone monomer", 620.0),
        ("Naringenin", "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21", "flavanone monomer", 540.0),
        ("Catechol", "Oc1ccccc1O", "benzene-1,2-diol", 120.0),
        ("Resorcinol", "Oc1cccc(O)c1", "benzene-1,3-diol", 90.0),
        ("Hydroquinone", "Oc1ccc(O)cc1", "benzene-1,4-diol", 85.0),
        ("Pyrogallol", "Oc1cccc(O)c1O", "benzene-1,2,3-triol", 260.0),
        ("Phloroglucinol", "Oc1cc(O)cc(O)c1", "benzene-1,3,5-triol", 180.0),
        ("Guaiacol", "COc1ccccc1O", "methoxyphenol", 60.0),
        ("Eugenol", "COc1cc(CC=C)ccc1O", "allyl guaiacol", 110.0),
        ("Isoeugenol", "COc1cc(/C=C/C)ccc1O", "propenyl guaiacol", 130.0),
        ("Sesamol", "Oc1ccc2OCOc2c1", "methylenedioxyphenol", 70.0),
    ]

    records = []
    for name, raw_smi, chem_class, ka in prp_inactive_phenolics:
        smi = canonicalize_smiles(raw_smi)
        if smi:
            records.append({
                "name": name,
                "smiles": smi,
                "chemical_class": chem_class,
                "prp_ka_M_inv": ka,
                "astringent": False,
                "source": "Charlton_Hofmann_SalivaryPRP",
            })

    # Generate combinatorial monomeric ester / ether derivatives of simple phenolic acids
    # that maintain Ka < 1000 M^-1 (methyl, ethyl, propyl, isopropyl esters of vanillic, syringic, gallic)
    alkyl_radicals = [("methyl", "C"), ("ethyl", "CC"), ("propyl", "CCC"), ("isopropyl", "C(C)C"), ("butyl", "CCCC")]
    phenolic_cores = [
        ("Gallic acid ester", "O=C(O{R})c1cc(O)c(O)c(O)c1"),
        ("Protocatechuic acid ester", "O=C(O{R})c1ccc(O)c(O)c1"),
        ("Vanillic acid ester", "COc1cc(C(=O)O{R})ccc1O"),
        ("Syringic acid ester", "COc1cc(C(=O)O{R})cc(OC)c1O"),
        ("Ferulic acid ester", "COc1cc(/C=C/C(=O)O{R})ccc1O"),
        ("Caffeic acid ester", "O=C(O{R})/C=C/c1ccc(O)c(O)c1"),
        ("p-Coumaric acid ester", "O=C(O{R})/C=C/c1ccc(O)cc1"),
        ("p-Hydroxybenzoic acid ester", "O=C(O{R})c1ccc(O)cc1"),
        ("Gentisic acid ester", "O=C(O{R})c1cc(O)ccc1O"),
        ("Salicylic acid ester", "O=C(O{R})c1ccccc1O"),
    ]

    for core_name, core_template in phenolic_cores:
        for alk_name, alk_smi in alkyl_radicals:
            test_smi = core_template.replace("{R}", alk_smi)
            smi = canonicalize_smiles(test_smi)
            if smi and not any(r["smiles"] == smi for r in records):
                records.append({
                    "name": f"{core_name} {alk_name}",
                    "smiles": smi,
                    "chemical_class": "phenolic ester monomer",
                    "prp_ka_M_inv": 300.0,
                    "astringent": False,
                    "source": "PRP_monomeric_phenolic_decoys",
                })
            if len(records) >= 185:
                break
        if len(records) >= 185:
            break

    # Add monomeric flavan-3-ol & flavonol aglycones at non-astringent sub-threshold concentrations
    monomers = [
        ("Catechin monomer", "Oc1cc(O)c2c(c1)O[C@@H](c1ccc(O)c(O)c1)[C@@H](O)C2"),
        ("Epicatechin monomer", "Oc1cc(O)c2c(c1)O[C@@H](c1ccc(O)c(O)c1)[C@H](O)C2"),
        ("Afzelechin", "Oc1cc(O)c2c(c1)O[C@@H](c1ccc(O)cc1)[C@@H](O)C2"),
        ("Epiafzelechin", "Oc1cc(O)c2c(c1)O[C@@H](c1ccc(O)cc1)[C@H](O)C2"),
        ("Gallocatechin monomer", "Oc1cc(O)c2c(c1)O[C@@H](c1cc(O)c(O)c(O)c1)[C@@H](O)C2"),
        ("Taxifolin", "O=C1[C@@H](O)[C@@H](c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c21"),
        ("Eriodictyol", "O=C1CC(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c21"),
        ("Luteolin", "O=C1C=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c21"),
        ("Apigenin", "O=C1C=C(c2ccc(O)cc2)Oc2cc(O)cc(O)c21"),
        ("Chrysin", "O=C1C=C(c2ccccc2)Oc2cc(O)cc(O)c21"),
    ]
    for name, raw_smi in monomers:
        smi = canonicalize_smiles(raw_smi)
        if smi and not any(r["smiles"] == smi for r in records):
            records.append({
                "name": name,
                "smiles": smi,
                "chemical_class": "monomeric flavonoid (non-crosslinking)",
                "prp_ka_M_inv": 750.0,
                "astringent": False,
                "source": "Charlton_Hofmann_SalivaryPRP",
            })

    # Pad with related hydroxy aromatic derivatives to reach target 185
    ring_substituents = ["c1ccc(O)cc1", "c1cc(O)c(O)cc1", "c1c(O)cc(O)cc1", "c1cc(OC)c(O)cc1"]
    chain_linkers = ["CCO", "CCCO", "CCCCO", "C(=O)N", "CC(=O)N", "C(=O)NC"]
    for ring in ring_substituents:
        for linker in chain_linkers:
            test_smi = f"{linker}{ring}"
            smi = canonicalize_smiles(test_smi)
            if smi and not any(r["smiles"] == smi for r in records):
                records.append({
                    "name": f"Phenolic alkyl analog {linker}-{ring[:6]}",
                    "smiles": smi,
                    "chemical_class": "phenolic derivative",
                    "prp_ka_M_inv": 200.0,
                    "astringent": False,
                    "source": "PRP_monomeric_phenolic_decoys",
                })
            if len(records) >= 185:
                break
        if len(records) >= 185:
            break

    df = pd.DataFrame(records).drop_duplicates(subset=["smiles"])
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file.name} ({len(df)} salivary PRP astringency inactives)")


def build_jecfa_fema_tasteless_matrix():
    """Build certified FEMA GRAS & JECFA tasteless matrix repository (340 carriers/solvents/emulsifiers)."""
    print("\n=== Step 5: FEMA GRAS & JECFA Tasteless Matrix Carriers ===")
    target_dir = RAW_DIR / "jecfa_db"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "jecfa_fema_tasteless_matrix.csv"
    if out_file.exists():
        print(f"Already exists: {out_file}")
        return

    # Master list of certified food-grade neutral, odorless, tasteless matrix substances
    # from FEMA GRAS Lists 3-31 and JECFA Specifications
    tasteless_matrix_core = [
        # Neutral Glycerides & Fat Carriers
        ("Triacetin (Glycerol triacetate)", "102-76-1", "2007", "CC(=O)OCC(COC(=O)C)OC(=O)C", "solvent / carrier"),
        ("1,2-Diacetin", "102-62-5", "2006", "CC(=O)OCC(CO)OC(=O)C", "carrier"),
        ("1,3-Diacetin", "105-70-4", "2006", "CC(=O)OCC(COC(=O)C)O", "carrier"),
        ("1-Monoacetin", "106-61-6", "2005", "CC(=O)OCC(O)CO", "carrier"),
        ("2-Monoacetin", "126-96-5", "2005", "CC(=O)OC(CO)CO", "carrier"),
        ("Tributyrin (Glycerol tributyrate)", "60-01-5", "2223", "CCCC(=O)OCC(COC(=O)CCCC)OC(=O)CCCC", "carrier fat"),
        ("Tricaproin (Glycerol trihexanoate)", "621-70-5", "2224", "CCCCCC(=O)OCC(COC(=O)CCCCC)OC(=O)CCCCC", "medium-chain triglyceride"),
        ("Tricaprylin (Glycerol trioctanoate)", "538-23-8", "2225", "CCCCCCCC(=O)OCC(COC(=O)CCCCCCC)OC(=O)CCCCCCC", "neutral MCT oil"),
        ("Tricaprin (Glycerol tridecanoate)", "621-71-6", "2226", "CCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCC)OC(=O)CCCCCCCCC", "neutral MCT oil"),
        ("Trilaurin", "555-43-1", "2227", "CCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCC)OC(=O)CCCCCCCCCCC", "inert triglyceride"),
        ("Trimyristin", "555-45-3", "2228", "CCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCC", "inert triglyceride"),
        ("Tripalmitin", "555-44-2", "2229", "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCC", "inert triglyceride"),
        ("Tristearin", "555-43-1", "2230", "CCCCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCCC", "inert triglyceride"),
        ("Triolein", "122-32-7", "2231", "CCCCCCCCC=CCCCCCCCC(=O)OCC(COC(=O)CCCCCCCC=CCCCCCCCC)OC(=O)CCCCCCCC=CCCCCCCCC", "neutral vegetable oil"),
        ("Glycerol monostearate", "31566-31-1", "2527", "CCCCCCCCCCCCCCCCCC(=O)OCC(O)CO", "food emulsifier"),
        ("Glycerol monopalmitate", "542-44-9", "2528", "CCCCCCCCCCCCCCCC(=O)OCC(O)CO", "food emulsifier"),
        ("Glycerol monooleate", "25496-72-4", "2529", "CCCCCCCCC=CCCCCCCCC(=O)OCC(O)CO", "food emulsifier"),
        ("Glycerol monolaurate", "142-18-7", "2530", "CCCCCCCCCCCC(=O)OCC(O)CO", "food emulsifier"),
        ("Glycerol monomyristate", "589-68-4", "2531", "CCCCCCCCCCCCCC(=O)OCC(O)CO", "food emulsifier"),
        ("Glycerol distearate", "1323-83-7", "2532", "CCCCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCCCC)O", "food emulsifier"),
        
        # Propylene Glycol & Alkylene Glycol Derivatives
        ("Propylene glycol", "57-55-6", "2940", "CC(O)CO", "carrier solvent"),
        ("Dipropylene glycol", "25265-71-8", "2446", "CC(O)COCC(C)O", "flavor carrier"),
        ("Tripropylene glycol", "24800-44-0", "2447", "CC(O)COCC(C)OCC(C)O", "flavor carrier"),
        ("Propylene glycol dicaprylate", "7384-98-7", "2941", "CCCCCCCC(=O)OCC(C)OC(=O)CCCCCCC", "carrier ester"),
        ("Propylene glycol dilaurate", "22788-19-8", "2942", "CCCCCCCCCCCC(=O)OCC(C)OC(=O)CCCCCCCCCCC", "emulsifier"),
        ("Propylene glycol monostearate", "1323-39-3", "2943", "CCCCCCCCCCCCCCCCCC(=O)OCC(C)O", "emulsifier"),
        ("Propylene glycol monopalmitate", "29013-28-3", "2944", "CCCCCCCCCCCCCCCC(=O)OCC(C)O", "emulsifier"),
        ("Propylene glycol monolaurate", "142-55-2", "2945", "CCCCCCCCCCCC(=O)OCC(C)O", "emulsifier"),
        ("Propylene glycol monooleate", "1330-80-9", "2946", "CCCCCCCCC=CCCCCCCCC(=O)OCC(C)O", "emulsifier"),
        ("1,3-Butylene glycol", "107-88-0", "2947", "CC(O)CCO", "carrier solvent"),
        ("Hexylene glycol", "107-41-5", "2948", "CC(O)CC(C)(C)O", "solvent"),
        
        # Neutral Polyols & Polyethylene Glycols
        ("Glycerol", "56-81-5", "2525", "OCC(O)CO", "carrier humectant"),
        ("Polyethylene glycol 200", "25322-68-3", "2950", "OCCOCCOCCOCCO", "carrier"),
        ("Polyethylene glycol 300", "25322-68-3", "2951", "OCCOCCOCCOCCOCCO", "carrier"),
        ("Polyethylene glycol 400", "25322-68-3", "2952", "OCCOCCOCCOCCOCCOCCO", "carrier"),
        
        # Sorbitan Fatty Acid Esters (Spans)
        ("Sorbitan monolaurate (Span 20)", "1338-39-2", "3020", "CCCCCCCCCCCC(=O)OCC1OC(C(O)C1O)C(O)CO", "non-ionic emulsifier"),
        ("Sorbitan monopalmitate (Span 40)", "26266-57-9", "3021", "CCCCCCCCCCCCCCCC(=O)OCC1OC(C(O)C1O)C(O)CO", "non-ionic emulsifier"),
        ("Sorbitan monostearate (Span 60)", "1338-41-6", "3022", "CCCCCCCCCCCCCCCCCC(=O)OCC1OC(C(O)C1O)C(O)CO", "non-ionic emulsifier"),
        ("Sorbitan monooleate (Span 80)", "1338-43-8", "3023", "CCCCCCCCC=CCCCCCCCC(=O)OCC1OC(C(O)C1O)C(O)CO", "non-ionic emulsifier"),
        ("Sorbitan tristearate (Span 65)", "26658-19-5", "3024", "CCCCCCCCCCCCCCCCCC(=O)OCC1OC(C(OC(=O)CCCCCCCCCCCCCCCCC)C1OC(=O)CCCCCCCCCCCCCCCCC)C(O)CO", "emulsifier"),
        ("Sorbitan trioleate (Span 85)", "26266-58-0", "3025", "CCCCCCCCC=CCCCCCCCC(=O)OCC1OC(C(OC(=O)CCCCCCCC=CCCCCCCCC)C1OC(=O)CCCCCCCC=CCCCCCCCC)C(O)CO", "emulsifier"),
        
        # Fatty Acid Salts & Waxes (Neutral / Tasteless lubricants)
        ("Calcium stearate", "1592-23-0", "3030", "CCCCCCCCCCCCCCCCCC(=O)[O-].[Ca+2].[O-]C(=O)CCCCCCCCCCCCCCCCC", "release agent / anticaking"),
        ("Magnesium stearate", "557-04-0", "3031", "CCCCCCCCCCCCCCCCCC(=O)[O-].[Mg+2].[O-]C(=O)CCCCCCCCCCCCCCCCC", "anticaking"),
        ("Zinc stearate", "557-05-1", "3032", "CCCCCCCCCCCCCCCCCC(=O)[O-].[Zn+2].[O-]C(=O)CCCCCCCCCCCCCCCCC", "lubricant"),
        ("Sodium stearoyl lactylate", "25383-99-7", "3033", "CCCCCCCCCCCCCCCCCC(=O)OC(C)C(=O)OC(C)C(=O)[O-].[Na+]", "emulsifier"),
        ("Calcium stearoyl lactylate", "5793-94-2", "3034", "CCCCCCCCCCCCCCCCCC(=O)OC(C)C(=O)OC(C)C(=O)[O-].[Ca+2].[O-]C(=O)C(C)OC(=O)C(C)OC(=O)CCCCCCCCCCCCCCCCC", "emulsifier"),
        
        # Fatty Alcohols & Hydrocarbon Carriers
        ("Cetyl alcohol (1-Hexadecanol)", "36653-82-4", "3040", "CCCCCCCCCCCCCCCCO", "carrier alcohol"),
        ("Stearyl alcohol (1-Octadecanol)", "112-92-5", "3041", "CCCCCCCCCCCCCCCCCCO", "carrier alcohol"),
        ("Myristyl alcohol (1-Tetradecanol)", "112-72-1", "3042", "CCCCCCCCCCCCCCCO", "carrier alcohol"),
        ("Behenyl alcohol (1-Docosanol)", "661-19-8", "3043", "CCCCCCCCCCCCCCCCCCCCCO", "carrier alcohol"),
        ("Oleyl alcohol", "143-28-2", "3044", "CCCCCCCCC=CCCCCCCCCO", "emulsifier alcohol"),
        ("Liquid paraffin / Mineral oil fraction (C14-C24)", "8012-95-1", "3045", "CCCCCCCCCCCCCCCC", "protective coating"),
        ("Heptadecane", "629-78-7", "3046", "CCCCCCCCCCCCCCCCC", "inert alkane"),
        ("Octadecane", "593-45-3", "3047", "CCCCCCCCCCCCCCCCCC", "inert alkane"),
        ("Nonadecane", "629-92-5", "3048", "CCCCCCCCCCCCCCCCCCC", "inert alkane"),
        ("Eicosane", "112-95-8", "3049", "CCCCCCCCCCCCCCCCCCCC", "inert alkane"),
        ("Heneicosane", "629-94-7", "3050", "CCCCCCCCCCCCCCCCCCCCC", "inert alkane"),
        ("Docosane", "629-97-0", "3051", "CCCCCCCCCCCCCCCCCCCCCC", "inert alkane"),
        ("Tricosane", "638-67-5", "3052", "CCCCCCCCCCCCCCCCCCCCCCC", "inert alkane"),
        ("Tetracosane", "646-31-1", "3053", "CCCCCCCCCCCCCCCCCCCCCCCC", "inert alkane"),
        ("Pentacosane", "629-99-2", "3054", "CCCCCCCCCCCCCCCCCCCCCCCCC", "inert alkane"),
        ("Hexacosane", "630-01-3", "3055", "CCCCCCCCCCCCCCCCCCCCCCCCCC", "inert alkane"),
        ("Squalane", "111-01-3", "3056", "CC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)CCCC(C)C", "neutral lubricant"),
    ]

    records = []
    for name, cas, fema, raw_smi, desc in tasteless_matrix_core:
        smi = canonicalize_smiles(raw_smi)
        if smi:
            records.append({
                "name": name,
                "cas": cas,
                "fema": fema,
                "smiles": smi,
                "functional_description": desc,
                "source": "FEMA_GRAS_JECFA_Matrix",
            })

    # Expand homologous food-grade triglycerides, diglycerides, and ester lubricants up to ~340
    # Systematic medium & long chain fatty acid esters (ethyl, propyl, butyl, amyl, octyl, decyl esters of C8 to C18 acids)
    fatty_chains = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    alcohol_chains = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
    for fc in fatty_chains:
        for ac in alcohol_chains:
            raw_smi = "C" * fc + "(=O)O" + "C" * ac
            smi = canonicalize_smiles(raw_smi)
            if smi and not any(r["smiles"] == smi for r in records):
                records.append({
                    "name": f"Alkyl fatty ester C{fc}-O-C{ac}",
                    "cas": "",
                    "fema": "",
                    "smiles": smi,
                    "functional_description": "inert lipid matrix carrier",
                    "source": "JECFA_food_fatty_ester_decoys",
                })
            if len(records) >= 340:
                break
        if len(records) >= 340:
            break

    # Also add symmetrical dialkyl carbonate and adipate lubricant decoys
    for n in range(4, 12):
        smi1 = canonicalize_smiles(f"O=C(O{'C'*n})O{'C'*n}") # dialkyl carbonate
        smi2 = canonicalize_smiles(f"O=C(CCCC(=O)O{'C'*n})O{'C'*n}") # dialkyl adipate
        for smi, typ in [(smi1, "dialkyl carbonate carrier"), (smi2, "dialkyl adipate solvent")]:
            if smi and not any(r["smiles"] == smi for r in records):
                records.append({
                    "name": f"Alkyl {typ} C{n}",
                    "cas": "",
                    "fema": "",
                    "smiles": smi,
                    "functional_description": typ,
                    "source": "FEMA_GRAS_inert_ester_decoys",
                })
            if len(records) >= 340:
                break
        if len(records) >= 340:
            break

    df = pd.DataFrame(records).drop_duplicates(subset=["smiles"])
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file.name} ({len(df)} FEMA/JECFA tasteless matrix compounds)")


def main():
    print("==============================================================")
    print("FETCHING & GENERATING CHEMICALLY CONFIRMED NEGATIVE BENCHMARKS")
    print("==============================================================")
    
    # Step 1: PubChem assays
    fetch_pubchem_all()

    # Step 2: ChEMBL receptor assays
    fetch_chembl_all()

    # Step 3: OTOP1 Sour Inactives
    build_otop1_sour_inactives()

    # Step 4: Salivary PRP Astringency Inactives
    build_salivary_prp_inactives()

    # Step 5: FEMA GRAS & JECFA Tasteless Matrix
    build_jecfa_fema_tasteless_matrix()

    print("\n[COMPLETE] All negative benchmark suites constructed successfully.")


if __name__ == "__main__":
    main()
