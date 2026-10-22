"""
FLAVRI: Unified Multimodal Chemosensory Foundation Model (FLAVRI 0.8)
148-Dimensional Olfactory, Gustatory, and Chemesthesis Architecture
"""

from flavri.constants import (
    NUM_CHANNELS,
    NUM_TASTE,
    NUM_ODOR,
    ALL_CHANNELS,
    TASTE_CHANNELS,
    ODOR_CHANNELS,
    CANONICAL_ODOR_DESCRIPTORS,
    ODOR_DESCRIPTOR_MAP,
)
from flavri.models.model import FlavriMultimodalModel
from flavri.featurizer import mol2graph

__version__ = "0.8.0"

__all__ = [
    "__version__",
    "FlavriMultimodalModel",
    "mol2graph",
    "NUM_CHANNELS",
    "NUM_TASTE",
    "NUM_ODOR",
    "ALL_CHANNELS",
    "TASTE_CHANNELS",
    "ODOR_CHANNELS",
    "CANONICAL_ODOR_DESCRIPTORS",
    "ODOR_DESCRIPTOR_MAP",
]
