"""
parse_pairs.py
Exhaustive extraction of empirical pairwise interaction mixtures and biophysically grounded additive neutrality pairs (FLAVRI_TASTE_PAIRS).
Includes:
- Pure chemical TPDB synergy records & cloned TAS2R bitter masking challenges
- Sub-pair resolution for multi-component (MSG, NaCl) rows
- Allosteric Kokumi CaSR receptor synergy pairs
- Psychophysical cross-modal interactions (heat-suppression, tingling-salivation)
- Concentration ratios for continuous titration matrices (Yamaguchi, Keast & Breslin)
- Balanced biophysical additive neutrality anchor pairs (Delta = 0, Mask = 1)
"""

import math
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

from utils import (
    CHANNELS,
    canonicalize_smiles,
    parse_peptide_sequence,
)

RAW_ROOT = Path(__file__).resolve().parent.parent / "exploration" / "raw"
DATA_COLLECTION = Path(__file__).resolve().parent.parent


def get_tpdb_derivatives_map() -> Dict[str, str]:
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


def canonical_pair_order(smi_a: str, smi_b: str, conc_a: float = 0.5, conc_b: float = 0.5) -> Tuple[str, str, float, float]:
    """Enforce strict commutative ordering: a < b, aligning concentrations accordingly."""
    if smi_a < smi_b:
        return smi_a, smi_b, conc_a, conc_b
    else:
        return smi_b, smi_a, conc_b, conc_a


def parse_tpdb_synergy_pairs(deriv_dict: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Dataset 1: TastePeptides Synergy.
    Filters strictly for pure chemical agonists and cloned receptor challenges:
    - Pure single compounds: MSG, NaCl, IMP, GMP, Sucrose, Quinine
    - Method A: Multi-component pure rows (MSG, NaCl) resolved to their biophysical target agonist
    - Cell-based bitter receptor challenges (HEK 293T hTAS2R16 / hTAS2R4 / hTAS2R14) paired with reference challenge ligands
    """
    path = RAW_ROOT / "tastepeptides_db" / "1-tastepeptides_synergy-2024-0621.csv"
    pairs = []
    if not path.exists():
        return pairs

    partner_smiles = {
        "msg": canonicalize_smiles("[Na+].[O-]C(=O)CCC(N)C(=O)O"),
        "nacl": canonicalize_smiles("[Na+].[Cl-]"),
        "saline": canonicalize_smiles("[Na+].[Cl-]"),
        "imp": canonicalize_smiles("O=P(O)(O)OCC1OC(C(O)C1O)n1cnc2c1nc[nH]c2=O"),
        "gmp": canonicalize_smiles("Nc1nc2c(ncn2C2OC(COP(=O)(O)O)C(O)C2O)c(=O)[nH]1"),
        "sucrose": canonicalize_smiles("C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O"),
        "quinine": canonicalize_smiles("COC1=CC2=C(C=CN=C2C=C1)C(C3CC4CCN3CC4C=C)O"),
        "caffeine": canonicalize_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        "tryptophan": canonicalize_smiles("N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
        "isoleucine": canonicalize_smiles("CC[C@H](C)[C@H](N)C(=O)O"),
    }

    salicin_smi = canonicalize_smiles("OCC1OC(Oc2ccccc2CO)C(O)C(O)C1O")
    quinine_smi = partner_smiles["quinine"]

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        partner_raw = str(row.get("Modulating Object", "")).strip().lower()
        pep_name = str(row.get("PepName", "")).strip()
        smi_a = parse_peptide_sequence(pep_name, deriv_dict)
        if not smi_a:
            continue

        prop = str(row.get("Modulating Properties", "")).strip().lower()
        targets = []

        # 1. Pure chemical agonist matches
        if partner_raw in partner_smiles:
            targets.append((partner_smiles[partner_raw], f"TPDB_Synergy_{partner_raw}"))

        # 2. Method A: Multi-component pure rows (MSG, NaCl)
        elif "msg, nacl" in partner_raw:
            if "umami-enhancing" in prop:
                targets.append((partner_smiles["msg"], "TPDB_Synergy_MSG_Resolved"))
            elif "salt-enhancing" in prop:
                targets.append((partner_smiles["nacl"], "TPDB_Synergy_NaCl_Resolved"))
            elif "bitter-masking" in prop or "bitterness-masking" in prop:
                targets.append((partner_smiles["nacl"], "TPDB_Synergy_NaCl_BitterMasking"))
            elif "thickness" in prop or "mouthfulness" in prop:
                targets.append((partner_smiles["msg"], "TPDB_Synergy_MSG_KokumiThickness"))

        # 3. Specific complex pure mixtures (e.g. caffeine, quinine, tryptophan)
        elif "caffeine, quinine" in partner_raw:
            targets.append((partner_smiles["quinine"], "TPDB_BitterMasking_Quinine"))
            targets.append((partner_smiles["caffeine"], "TPDB_BitterMasking_Caffeine"))
        elif "isoleucine" in partner_raw or "异亮氨酸" in partner_raw:
            targets.append((partner_smiles["isoleucine"], "TPDB_BitterMasking_Ile"))

        # 4. Cloned TAS2R bitter masking challenges
        elif "tas2r16" in partner_raw or "t2r16" in partner_raw:
            targets.append((salicin_smi, "TPDB_TAS2R16_Salicin_Inhibition"))
        elif "t2r4" in partner_raw or "t2r14" in partner_raw:
            targets.append((quinine_smi, "TPDB_TAS2R_Quinine_Inhibition"))

        for smi_b, source_tag in targets:
            if not smi_b or smi_a == smi_b:
                continue

            a, b, w_a, w_b = canonical_pair_order(smi_a, smi_b, 0.5, 0.5)

            deltas = {c: 0.0 for c in CHANNELS}
            masks = {c: 0 for c in CHANNELS}

            if "umami-enhancing" in prop:
                deltas["umami"] = 0.50
                masks["umami"] = 1
            elif "salt-enhancing" in prop:
                deltas["salty"] = 0.50
                masks["salty"] = 1
            elif "kokumi-enhancing" in prop or "thickness" in prop or "mouthfulness" in prop:
                deltas["kokumi"] = 0.50
                masks["kokumi"] = 1
            elif "bitter-masking" in prop or "bitterness-masking" in prop or "inhibition" in source_tag:
                deltas["bitter"] = -0.50
                masks["bitter"] = 1
            elif "sour-masking" in prop:
                deltas["sour"] = -0.50
                masks["sour"] = 1
            elif "sweet-enhancing" in prop:
                deltas["sweet"] = 0.50
                masks["sweet"] = 1

            for neutral_ch in ["cooling", "heat", "pungency", "tingling"]:
                deltas[neutral_ch] = 0.0
                masks[neutral_ch] = 1

            pairs.append({
                "smiles_a": a,
                "smiles_b": b,
                "ratio_a": w_a,
                "ratio_b": w_b,
                "deltas": deltas,
                "masks": masks,
                "source": source_tag
            })

    print(f"[TPDB Synergy] Parsed {len(pairs)} pure chemical & receptor interaction pairs.")
    return pairs


def parse_yamaguchi_pairs() -> List[Dict[str, Any]]:
    """Dataset 2: Yamaguchi Umami Synergy Matrix (Canonical 1:1 equimolar MSG x IMP pair)."""
    path = RAW_ROOT / "psychophysical_mixtures" / "yamaguchi_umami_synergy_matrix.csv"
    pairs = []
    if not path.exists():
        return pairs

    df = pd.read_csv(path)
    if len(df) == 0:
        return pairs

    row = df.iloc[0]
    smi_a = canonicalize_smiles(row.get("component_a_smiles"))
    smi_b = canonicalize_smiles(row.get("component_b_smiles"))
    if smi_a and smi_b and smi_a != smi_b:
        a, b, r_a, r_b = canonical_pair_order(smi_a, smi_b, 0.5, 0.5)
        # Peak synergistic potential at equimolar ratio (maximum cooperativity)
        deltas = {c: 0.0 for c in CHANNELS}
        masks = {c: 1 for c in CHANNELS}
        deltas["umami"] = 1.0

        pairs.append({
            "smiles_a": a,
            "smiles_b": b,
            "ratio_a": 0.5,
            "ratio_b": 0.5,
            "deltas": deltas,
            "masks": masks,
            "source": "Yamaguchi_Umami_Synergy_1to1"
        })

    print(f"[Yamaguchi Matrix] Parsed {len(pairs)} canonical 1:1 equimolar pair.")
    return pairs


def parse_keast_breslin_pairs() -> List[Dict[str, Any]]:
    """Dataset 3: Keast & Breslin Sweet-Sour Mutual Suppression Matrix (Canonical 1:1 equimolar pair)."""
    path = RAW_ROOT / "psychophysical_mixtures" / "keast_breslin_sweet_sour_matrix.csv"
    pairs = []
    if not path.exists():
        return pairs

    df = pd.read_csv(path)
    if len(df) == 0:
        return pairs

    row = df.iloc[0]
    smi_a = canonicalize_smiles(row.get("component_a_smiles"))
    smi_b = canonicalize_smiles(row.get("component_b_smiles"))
    if smi_a and smi_b and smi_a != smi_b:
        a, b, r_a, r_b = canonical_pair_order(smi_a, smi_b, 0.5, 0.5)
        # Mutual sweet-sour suppression at equimolar ratio
        deltas = {c: 0.0 for c in CHANNELS}
        masks = {c: 1 for c in CHANNELS}
        deltas["sweet"] = -0.30
        deltas["sour"] = -0.30

        pairs.append({
            "smiles_a": a,
            "smiles_b": b,
            "ratio_a": 0.5,
            "ratio_b": 0.5,
            "deltas": deltas,
            "masks": masks,
            "source": "Keast_Breslin_Sweet_Sour_1to1"
        })

    print(f"[Keast & Breslin] Parsed {len(pairs)} canonical 1:1 equimolar pair.")
    return pairs


def parse_breslin_beauchamp_pairs() -> List[Dict[str, Any]]:
    """Dataset 4: Breslin & Beauchamp Sodium-Induced Bitter Masking Matrix."""
    path = RAW_ROOT / "psychophysical_mixtures" / "breslin_beauchamp_sodium_bitter_matrix.csv"
    pairs = []
    if not path.exists():
        return pairs

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        smi_a = canonicalize_smiles(row.get("salt_smiles"))
        smi_b = canonicalize_smiles(row.get("bitter_smiles"))
        if not smi_a or not smi_b or smi_a == smi_b:
            continue

        a, b, r_a, r_b = canonical_pair_order(smi_a, smi_b, 0.5, 0.5)
        suppressed = bool(row.get("suppressed_by_salt", True))

        deltas = {c: 0.0 for c in CHANNELS}
        masks = {c: 0 for c in CHANNELS}

        deltas["bitter"] = -0.50 if suppressed else 0.0
        masks["bitter"] = 1

        for ch in ["sweet", "sour", "umami", "kokumi", "astringent", "cooling", "heat", "pungency", "tingling"]:
            deltas[ch] = 0.0
            masks[ch] = 1

        pairs.append({
            "smiles_a": a,
            "smiles_b": b,
            "ratio_a": r_a,
            "ratio_b": r_b,
            "deltas": deltas,
            "masks": masks,
            "source": "Breslin_Beauchamp_Sodium_Bitter"
        })

    print(f"[Breslin & Beauchamp] Parsed {len(pairs)} benchmark pairs.")
    return pairs


def generate_kokumi_allosteric_pairs(singles_df: pd.DataFrame, num_target: int = 60) -> List[Dict[str, Any]]:
    """
    Method B: Kokumi Allosteric Receptor Synergy.
    CaSR is an obligate allosteric receptor requiring an orthosteric cation/MSG base + kokumi peptide.
    """
    random.seed(42)
    pairs = []

    # Get verified kokumi peptides (gamma-glutamyls, GSH, CaSR agonists)
    kokumi_df = singles_df[(singles_df["y_kokumi"] >= 0.7) & (singles_df["mask_kokumi"] == 1)]
    kokumi_smiles = kokumi_df["smiles"].tolist()

    # Reference mineral and umami co-activators
    covalent_bases = [
        canonicalize_smiles("[Na+].[O-]C(=O)CCC(N)C(=O)O"),  # MSG
        canonicalize_smiles("[Cl-].[Na+]"),                  # NaCl
        canonicalize_smiles("[Cl-].[Cl-].[Ca+2]"),           # CaCl2
    ]
    covalent_bases = [b for b in covalent_bases if b]

    if not kokumi_smiles or not covalent_bases:
        return pairs

    used_pairs = set()
    for _ in range(num_target):
        k_smi = random.choice(kokumi_smiles)
        base_smi = random.choice(covalent_bases)
        if k_smi == base_smi:
            continue

        a, b, r_a, r_b = canonical_pair_order(k_smi, base_smi, 0.5, 0.5)
        if (a, b) in used_pairs:
            continue
        used_pairs.add((a, b))

        deltas = {c: 0.0 for c in CHANNELS}
        masks = {c: 0 for c in CHANNELS}

        deltas["kokumi"] = 0.50
        masks["kokumi"] = 1

        for ch in ["cooling", "heat", "pungency", "tingling"]:
            deltas[ch] = 0.0
            masks[ch] = 1

        pairs.append({
            "smiles_a": a,
            "smiles_b": b,
            "ratio_a": r_a,
            "ratio_b": r_b,
            "deltas": deltas,
            "masks": masks,
            "source": "Biophysical_CaSR_Kokumi_Synergy"
        })

    print(f"[Kokumi Synergy] Generated {len(pairs)} CaSR allosteric synergy pairs.")
    return pairs


def generate_cross_modal_pairs(singles_df: pd.DataFrame, num_target: int = 40) -> List[Dict[str, Any]]:
    """
    Method C: Psychophysical Cross-Modal Trigeminal Interactions.
    1. Thermal burn (TRPV1) mutual suppression of sweetness and umami (Prescott et al.)
    2. Tingling alkylamides (sanshool) salivation enhancement of salt
    """
    random.seed(42)
    pairs = []

    heat_smiles = singles_df[singles_df["y_heat"] > 0.6]["smiles"].tolist()
    tingling_smiles = singles_df[singles_df["y_tingling"] > 0.6]["smiles"].tolist()
    sweet_smiles = singles_df[singles_df["y_sweet"] > 0.8]["smiles"].tolist()
    umami_smiles = singles_df[singles_df["y_umami"] > 0.8]["smiles"].tolist()
    salt_smiles = singles_df[singles_df["y_salty"] > 0.8]["smiles"].tolist()

    used = set()

    # 1. Thermal Burn (Heat) suppresses Sweet
    if heat_smiles and sweet_smiles:
        for _ in range(num_target // 3):
            h = random.choice(heat_smiles)
            sw = random.choice(sweet_smiles)
            a, b, r_a, r_b = canonical_pair_order(h, sw, 0.5, 0.5)
            if (a, b) in used:
                continue
            used.add((a, b))
            deltas = {c: 0.0 for c in CHANNELS}
            masks = {c: 0 for c in CHANNELS}
            deltas["sweet"] = -0.30
            masks["sweet"] = 1
            for neutral in ["cooling", "tingling"]:
                deltas[neutral] = 0.0
                masks[neutral] = 1
            pairs.append({
                "smiles_a": a, "smiles_b": b, "ratio_a": r_a, "ratio_b": r_b,
                "deltas": deltas, "masks": masks, "source": "CrossModal_Heat_Sweet_Suppression"
            })

    # 2. Thermal Burn (Heat) suppresses Umami
    if heat_smiles and umami_smiles:
        for _ in range(num_target // 3):
            h = random.choice(heat_smiles)
            um = random.choice(umami_smiles)
            a, b, r_a, r_b = canonical_pair_order(h, um, 0.5, 0.5)
            if (a, b) in used:
                continue
            used.add((a, b))
            deltas = {c: 0.0 for c in CHANNELS}
            masks = {c: 0 for c in CHANNELS}
            deltas["umami"] = -0.30
            masks["umami"] = 1
            for neutral in ["cooling", "tingling"]:
                deltas[neutral] = 0.0
                masks[neutral] = 1
            pairs.append({
                "smiles_a": a, "smiles_b": b, "ratio_a": r_a, "ratio_b": r_b,
                "deltas": deltas, "masks": masks, "source": "CrossModal_Heat_Umami_Suppression"
            })

    # 3. Tingling (Sanshool) enhances Saltiness perception
    if tingling_smiles and salt_smiles:
        for _ in range(num_target // 3):
            ti = random.choice(tingling_smiles)
            sa = random.choice(salt_smiles)
            a, b, r_a, r_b = canonical_pair_order(ti, sa, 0.5, 0.5)
            if (a, b) in used:
                continue
            used.add((a, b))
            deltas = {c: 0.0 for c in CHANNELS}
            masks = {c: 0 for c in CHANNELS}
            deltas["salty"] = 0.35
            masks["salty"] = 1
            for neutral in ["cooling", "heat"]:
                deltas[neutral] = 0.0
                masks[neutral] = 1
            pairs.append({
                "smiles_a": a, "smiles_b": b, "ratio_a": r_a, "ratio_b": r_b,
                "deltas": deltas, "masks": masks, "source": "CrossModal_Tingling_Salt_Enhancement"
            })

    print(f"[Cross-Modal] Generated {len(pairs)} psychophysical cross-modal interaction pairs.")
    return pairs


def generate_biophysical_neutral_pairs(singles_df: pd.DataFrame, num_target: int = 500) -> List[Dict[str, Any]]:
    """Method D: Grounded additive neutrality anchor pairs (delta = 0 across all channels, mask = 1)."""
    random.seed(42)
    neutral_pairs = []

    tasteless_mask = (singles_df["mask_sweet"] == 1) & (singles_df["y_sweet"] == 0.0) & \
                     (singles_df["mask_bitter"] == 1) & (singles_df["y_bitter"] == 0.0) & \
                     (singles_df["mask_umami"] == 1) & (singles_df["y_umami"] == 0.0)
    tasteless_smiles = singles_df[tasteless_mask]["smiles"].tolist()

    active_mask = (singles_df["y_sweet"] > 0.5) | (singles_df["y_bitter"] > 0.5) | (singles_df["y_umami"] > 0.5)
    active_smiles = singles_df[active_mask]["smiles"].tolist()

    cooling_smiles = singles_df[singles_df["y_cooling"] > 0.5]["smiles"].tolist()
    salt_smiles = singles_df[singles_df["y_salty"] > 0.5]["smiles"].tolist()
    heat_smiles = singles_df[singles_df["y_heat"] > 0.5]["smiles"].tolist()
    sweet_smiles = singles_df[singles_df["y_sweet"] > 0.5]["smiles"].tolist()

    used_pairs = set()

    # 1. Tasteless x Active Tastant
    if tasteless_smiles and active_smiles:
        for _ in range(num_target // 2):
            t_smi = random.choice(tasteless_smiles)
            a_smi = random.choice(active_smiles)
            if t_smi == a_smi:
                continue
            a, b, r_a, r_b = canonical_pair_order(t_smi, a_smi, 0.5, 0.5)
            if (a, b) in used_pairs:
                continue
            used_pairs.add((a, b))
            neutral_pairs.append({
                "smiles_a": a,
                "smiles_b": b,
                "ratio_a": r_a,
                "ratio_b": r_b,
                "deltas": {c: 0.0 for c in CHANNELS},
                "masks": {c: 1 for c in CHANNELS},
                "source": "Neutral_Tasteless_Anchor"
            })

    # 2. Orthogonal Cooling x Salt
    if cooling_smiles and salt_smiles:
        for _ in range(num_target // 4):
            c_smi = random.choice(cooling_smiles)
            s_smi = random.choice(salt_smiles)
            if c_smi == s_smi:
                continue
            a, b, r_a, r_b = canonical_pair_order(c_smi, s_smi, 0.5, 0.5)
            if (a, b) in used_pairs:
                continue
            used_pairs.add((a, b))
            neutral_pairs.append({
                "smiles_a": a,
                "smiles_b": b,
                "ratio_a": r_a,
                "ratio_b": r_b,
                "deltas": {c: 0.0 for c in CHANNELS},
                "masks": {c: 1 for c in CHANNELS},
                "source": "Neutral_Orthogonal_Cooling_Salt"
            })

    # 3. Orthogonal Heat x Sweet
    if heat_smiles and sweet_smiles:
        for _ in range(num_target // 4):
            h_smi = random.choice(heat_smiles)
            sw_smi = random.choice(sweet_smiles)
            if h_smi == sw_smi:
                continue
            a, b, r_a, r_b = canonical_pair_order(h_smi, sw_smi, 0.5, 0.5)
            if (a, b) in used_pairs:
                continue
            used_pairs.add((a, b))
            neutral_pairs.append({
                "smiles_a": a,
                "smiles_b": b,
                "ratio_a": r_a,
                "ratio_b": r_b,
                "deltas": {c: 0.0 for c in CHANNELS},
                "masks": {c: 1 for c in CHANNELS},
                "source": "Neutral_Orthogonal_Heat_Sweet"
            })

    print(f"[Neutrality Anchors] Generated {len(neutral_pairs)} additive neutral pairs.")
    return neutral_pairs


def ingest_all_pairs() -> pd.DataFrame:
    """Build and validate FLAVRI_TASTE_PAIRS."""
    deriv_dict = get_tpdb_derivatives_map()

    all_pairs = []
    all_pairs.extend(parse_tpdb_synergy_pairs(deriv_dict))
    all_pairs.extend(parse_yamaguchi_pairs())
    all_pairs.extend(parse_keast_breslin_pairs())
    all_pairs.extend(parse_breslin_beauchamp_pairs())

    singles_path = DATA_COLLECTION / "flavri_taste_singles.csv"
    if singles_path.exists():
        singles_df = pd.read_csv(singles_path)
        all_pairs.extend(generate_kokumi_allosteric_pairs(singles_df, num_target=75))
        all_pairs.extend(generate_cross_modal_pairs(singles_df, num_target=45))
        # Add balanced neutrality anchors
        num_non_neutral = len(all_pairs)
        all_pairs.extend(generate_biophysical_neutral_pairs(singles_df, num_target=num_non_neutral))

    rows = []
    for p in all_pairs:
        row = {
            "smiles_a": p["smiles_a"],
            "smiles_b": p["smiles_b"],
        }
        for ch in CHANNELS:
            row[f"delta_{ch}"] = p["deltas"][ch]
            row[f"mask_{ch}"] = p["masks"][ch]
        row["pair_source"] = p["source"]
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out = df_out.drop_duplicates(subset=["smiles_a", "smiles_b"])
    print(f"[Done] Total unique pairwise mixture training samples: {len(df_out)}")
    return df_out


if __name__ == "__main__":
    df = ingest_all_pairs()
    out_path = DATA_COLLECTION / "flavri_taste_pairs.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved FLAVRI_TASTE_PAIRS to {out_path}")
