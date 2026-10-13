"""
flavri.data package
"""

from flavri.data.dataset import FlavriSinglesDataset, FlavriPairsDataset
from flavri.data.collator import SinglesCollator, PairsCollator

__all__ = [
    "FlavriSinglesDataset",
    "FlavriPairsDataset",
    "SinglesCollator",
    "PairsCollator",
]
