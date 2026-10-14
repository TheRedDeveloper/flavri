"""
flavri.models package
"""

from flavri.models.dmpnn import DMPNNBackbone, MultiHeadAttentivePooling
AttentiveGraphPooling = MultiHeadAttentivePooling  # alias for backwards compatibility
from flavri.models.chem_head import ChemicalReconstructionHead
from flavri.models.heads import SingleMoleculeHead, CommutativeSynergyHead
from flavri.models.model import FlavriMultimodalModel

__all__ = [
    "DMPNNBackbone",
    "MultiHeadAttentivePooling",
    "AttentiveGraphPooling",
    "ChemicalReconstructionHead",
    "SingleMoleculeHead",
    "CommutativeSynergyHead",
    "FlavriMultimodalModel",
]
