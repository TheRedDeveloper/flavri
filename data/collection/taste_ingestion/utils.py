"""
utils.py
Core chemical and mathematical utilities for FLAVRI taste dataset ingestion.
"""

import math
from typing import Optional, Dict
from rdkit import Chem

# Canonical 11 channels
CHANNELS = [
    "sweet",
    "salty",
    "sour",
    "bitter",
    "umami",
    "kokumi",
    "astringent",
    "cooling",
    "heat",
    "pungency",
    "tingling",
]

THREE_TO_ONE = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
}


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit, preserving disconnected ionic salts."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def parse_peptide_sequence(name_or_seq: str, deriv_dict: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Parse a peptide representation into canonical linear SMILES.
    Handles:
    - Derivatized dictionary lookup
    - 3-letter hyphenated sequences (e.g., 'Gly-Leu-Glu')
    - 1-letter FASTA sequence (e.g., 'AD', 'KGDEESLA')
    """
    if not isinstance(name_or_seq, str) or not name_or_seq.strip():
        return None
    name_clean = name_or_seq.strip()
    if name_clean in ('/', 'None', 'nan', ''):
        return None

    # 1. Lookup in derivative dictionary
    if deriv_dict and name_clean in deriv_dict:
        s = canonicalize_smiles(deriv_dict[name_clean])
        if s:
            return s

    # 2. Check 3-letter hyphenated sequence
    parts = name_clean.split('-')
    if len(parts) > 1 and all(p in THREE_TO_ONE for p in parts):
        seq = ''.join(THREE_TO_ONE[p] for p in parts)
        try:
            mol = Chem.MolFromSequence(seq)
            if mol:
                return Chem.MolToSmiles(mol)
        except Exception:
            pass

    # 3. Check 1-letter sequence (standard uppercase amino acids)
    if all(c in THREE_TO_ONE.values() for c in name_clean):
        try:
            mol = Chem.MolFromSequence(name_clean)
            if mol:
                return Chem.MolToSmiles(mol)
        except Exception:
            pass

    # 4. Fallback: try direct SMILES parse in case it was a SMILES
    return canonicalize_smiles(name_clean)


def threshold_to_potency(thresh_val: float, unit: str = "mM", pT_mid: float = 0.0, slope: float = 1.0) -> float:
    """
    Convert sensory detection/recognition threshold to continuous [0, 1] potency.
    y = sigmoid((-log10(T_mM) - pT_mid) / slope)
    """
    try:
        val = float(thresh_val)
        if val <= 0:
            return 1.0
        if unit.lower() in ("um", "umol/l", "umol_l", "micromolar"):
            val_mM = val * 1e-3
        else:
            val_mM = val
        
        pT = -math.log10(val_mM)
        z = (pT - pT_mid) / slope
        # Logistic sigmoid clamped to [0.0, 1.0]
        potency = 1.0 / (1.0 + math.exp(-z))
        return max(0.0, min(1.0, potency))
    except Exception:
        return 1.0


def affinity_to_potency(affinity_val: float, unit: str = "nM", pEC_min: float = 4.0, pEC_max: float = 9.0) -> float:
    """
    Convert continuous EC50 / IC50 (nM or uM) to [0, 1] potency via pEC50.
    """
    try:
        val = float(affinity_val)
        if val <= 0:
            return 0.0
        if unit.lower() == "um":
            val_M = val * 1e-6
        elif unit.lower() == "nm":
            val_M = val * 1e-9
        else:
            val_M = val
        
        pEC50 = -math.log10(val_M)
        scaled = (pEC50 - pEC_min) / (pEC_max - pEC_min)
        return max(0.0, min(1.0, scaled))
    except Exception:
        return 0.5
