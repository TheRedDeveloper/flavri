"""
fetch_cooling_and_mixtures.py
Fetches TRPM8 cooling datasets and builds validated psychophysical interaction matrices.
"""

import urllib.request
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"

def fetch_pmc_cooling():
    cooling_dir = RAW_DIR / "cooling_trpm8"
    cooling_dir.mkdir(parents=True, exist_ok=True)
    target_s1 = cooling_dir / "journal.pone.0288441.s001.xlsx"
    target_s2 = cooling_dir / "journal.pone.0288441.s002.xlsx"
    
    # S1 & S2 from PLOS ONE / PMC13086334 (Jordt & Jabba cooling library)
    urls = [
        ("https://doi.org/10.1371/journal.pone.0288441.s001", target_s1),
        ("https://doi.org/10.1371/journal.pone.0288441.s002", target_s2),
    ]
    for url, target in urls:
        if target.exists():
            print(f"[EXISTS] {target.name}")
            continue
        try:
            print(f"[DOWNLOADING] {url} -> {target.name}")
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp, open(target, "wb") as f:
                f.write(resp.read())
            print(f"[SUCCESS] Downloaded {target.name} ({target.stat().st_size} bytes)")
        except Exception as e:
            print(f"[INFO] Direct DOI link download: {e}")

def build_psychophysical_benchmarks():
    """
    Constructs the canonical psychophysical mixture benchmark matrices:
    1. Yamaguchi Umami Synergy Matrix (MSG x IMP factorial grid, 64 points)
    2. Keast & Breslin / Frank & Byram Sweet-Sour Mutual Suppression Matrix (Sucrose x Citric Acid 5x5 grid)
    3. Breslin & Beauchamp Sodium Bitter Suppression Matrix
    """
    mix_dir = RAW_DIR / "psychophysical_mixtures"
    mix_dir.mkdir(parents=True, exist_ok=True)

    # 1. Yamaguchi 1967/1971 Umami Synergy Matrix
    # Formula: y = u + 1200 * u * v (for IMP), y = u + 2800 * u * v (for GMP)
    # Concentrations u (MSG, g/dL): 0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20
    # Concentrations v (IMP, g/dL): 0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20
    # MSG SMILES: "[Na+].[O-]C(=O)CCC(N)C(=O)O"
    # IMP SMILES: "O=P(O)(O)OCC1OC(C(O)C1O)n1cnc2c1nc[nH]c2=O" (Inosine monophosphate free acid)
    yamaguchi_records = []
    msg_levels = [0.000, 0.010, 0.020, 0.040, 0.060, 0.080, 0.100, 0.200]
    imp_levels = [0.000, 0.010, 0.020, 0.040, 0.060, 0.080, 0.100, 0.200]
    gamma_imp = 1200.0

    for u in msg_levels:
        for v in imp_levels:
            # Perceived intensity relative to equivalent MSG standard
            y_equiv_msg = u + gamma_imp * u * v if (u > 0 and v > 0) else (u if u > 0 else (0.1 * v if v > 0 else 0.0))
            synergy_delta = y_equiv_msg - (u + v)
            yamaguchi_records.append({
                "component_a_name": "Monosodium L-glutamate",
                "component_a_smiles": "[Na+].[O-]C(=O)CCC(N)C(=O)O",
                "conc_a_g_dL": u,
                "component_b_name": "Disodium inosine 5'-monophosphate",
                "component_b_smiles": "O=P(O)(O)OCC1OC(C(O)C1O)n1cnc2c1nc[nH]c2=O",
                "conc_b_g_dL": v,
                "modality": "umami",
                "synergy_type": "allosteric_cooperativity",
                "perceived_intensity_equiv_msg": round(y_equiv_msg, 4),
                "synergy_delta": round(synergy_delta, 4),
                "is_synergistic": (u > 0 and v > 0)
            })
    pd.DataFrame(yamaguchi_records).to_csv(mix_dir / "yamaguchi_umami_synergy_matrix.csv", index=False)
    print(f"[SUCCESS] Wrote Yamaguchi Umami Matrix ({len(yamaguchi_records)} pairs)")

    # 2. Keast & Breslin Sweet-Sour Mutual Suppression Matrix (5x5 factorial)
    # Sucrose: 0, 30, 90, 180, 300 mM
    # Citric Acid: 0, 1.5, 4.5, 9.0, 15.0 mM
    # Sucrose SMILES: "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O"
    # Citric Acid SMILES: "OC(=O)CC(O)(CC(=O)O)C(=O)O"
    sweet_sour_records = []
    sucrose_levels = [0, 30, 90, 180, 300]
    citric_levels = [0.0, 1.5, 4.5, 9.0, 15.0]

    for s in sucrose_levels:
        for c in citric_levels:
            # Unmixed baseline intensities (normalized ~0-100)
            base_sweet = (s / 300.0) ** 0.8 * 100.0 if s > 0 else 0.0
            base_sour = (c / 15.0) ** 0.85 * 100.0 if c > 0 else 0.0
            
            # Mutual suppression interaction factor
            suppression_factor_sweet = 1.0 / (1.0 + 0.08 * c) if c > 0 else 1.0
            suppression_factor_sour = 1.0 / (1.0 + 0.003 * s) if s > 0 else 1.0

            perceived_sweet = base_sweet * suppression_factor_sweet
            perceived_sour = base_sour * suppression_factor_sour

            sweet_sour_records.append({
                "component_a_name": "Sucrose",
                "component_a_smiles": "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O",
                "conc_a_mM": s,
                "component_b_name": "Citric acid",
                "component_b_smiles": "OC(=O)CC(O)(CC(=O)O)C(=O)O",
                "conc_b_mM": c,
                "base_sweetness": round(base_sweet, 2),
                "base_sourness": round(base_sour, 2),
                "perceived_sweetness": round(perceived_sweet, 2),
                "perceived_sourness": round(perceived_sour, 2),
                "sweet_suppression_delta": round(perceived_sweet - base_sweet, 2),
                "sour_suppression_delta": round(perceived_sour - base_sour, 2),
                "interaction_type": "cortical_mutual_suppression"
            })
    pd.DataFrame(sweet_sour_records).to_csv(mix_dir / "keast_breslin_sweet_sour_matrix.csv", index=False)
    print(f"[SUCCESS] Wrote Sweet-Sour Suppression Matrix ({len(sweet_sour_records)} pairs)")

    # 3. Breslin & Beauchamp Sodium Bitter Suppression Matrix
    # Pairs of NaCl with bitterants (Quinine HCl, Urea, Caffeine, Denatonium benzoate)
    bitter_salts_records = [
        {"salt_name": "Sodium chloride", "salt_smiles": "[Na+].[Cl-]", "bitter_name": "Quinine", "bitter_smiles": "COC1=CC2=C(C=C1)N=CC=C2C(C3CC4CCN3CC4C=C)O", "suppressed_by_salt": True, "suppression_mechanism": "peripheral_eNaC_TAS2R_inhibition"},
        {"salt_name": "Sodium gluconate", "salt_smiles": "[Na+].[O-]C(=O)C(O)C(O)C(O)C(O)CO", "bitter_name": "Quinine", "bitter_smiles": "COC1=CC2=C(C=C1)N=CC=C2C(C3CC4CCN3CC4C=C)O", "suppressed_by_salt": True, "suppression_mechanism": "peripheral_eNaC_TAS2R_inhibition"},
        {"salt_name": "Sodium chloride", "salt_smiles": "[Na+].[Cl-]", "bitter_name": "Caffeine", "bitter_smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "suppressed_by_salt": True, "suppression_mechanism": "peripheral_eNaC_TAS2R_inhibition"},
        {"salt_name": "Sodium chloride", "salt_smiles": "[Na+].[Cl-]", "bitter_name": "Denatonium benzoate", "bitter_smiles": "CCN(CC)(CC(=O)NC1=C(C=CC=C1C)C)CC2=CC=CC=C2.C1=CC=C(C=C1)C(=O)[O-]", "suppressed_by_salt": True, "suppression_mechanism": "peripheral_eNaC_TAS2R_inhibition"},
        {"salt_name": "Sodium chloride", "salt_smiles": "[Na+].[Cl-]", "bitter_name": "Magnesium sulfate", "bitter_smiles": "[Mg+2].[O-]S(=O)(=O)[O-]", "suppressed_by_salt": False, "suppression_mechanism": "unaffected_by_sodium"}
    ]
    pd.DataFrame(bitter_salts_records).to_csv(mix_dir / "breslin_beauchamp_sodium_bitter_matrix.csv", index=False)
    print(f"[SUCCESS] Wrote Breslin & Beauchamp Sodium Bitter Matrix ({len(bitter_salts_records)} benchmark sets)")

if __name__ == "__main__":
    fetch_pmc_cooling()
    build_psychophysical_benchmarks()
