"""
flavri.constants
Unified 148-Dimensional Channel Taxonomy and Constants.
"""

from data.collection.odor_ingestion.constants import CANONICAL_ODOR_DESCRIPTORS

# 11 Canonical Taste / Trigeminal Channels
RAW_TASTE_CHANNELS = [
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

TASTE_CHANNELS = [f"taste_{t}" for t in RAW_TASTE_CHANNELS]

# Overlapping keys between taste and odor receive 'odor_' prefix:
# {"sweet", "sour", "bitter", "cooling"} -> {"odor_sweet", "odor_sour", "odor_bitter", "odor_cooling"}
OVERLAP_KEYS = set(RAW_TASTE_CHANNELS).intersection(set(CANONICAL_ODOR_DESCRIPTORS))

ODOR_DESCRIPTOR_MAP = {
    d: (f"odor_{d}" if d in OVERLAP_KEYS else d)
    for d in CANONICAL_ODOR_DESCRIPTORS
}

ODOR_CHANNELS = [ODOR_DESCRIPTOR_MAP[d] for d in CANONICAL_ODOR_DESCRIPTORS]

# Unified 148 channels: 11 taste + 137 odor
ALL_CHANNELS = TASTE_CHANNELS + ODOR_CHANNELS
assert len(ALL_CHANNELS) == 148, f"Expected 148 channels, got {len(ALL_CHANNELS)}"

CHANNEL_TO_IDX = {ch: i for i, ch in enumerate(ALL_CHANNELS)}
NUM_CHANNELS = len(ALL_CHANNELS)
NUM_TASTE = len(TASTE_CHANNELS)
NUM_ODOR = len(ODOR_CHANNELS)

TARGET_COLS_SINGLES = [f"y_{ch}" for ch in ALL_CHANNELS]
MASK_COLS_SINGLES = [f"mask_{ch}" for ch in ALL_CHANNELS]

TARGET_COLS_PAIRS = [f"delta_{ch}" for ch in ALL_CHANNELS]
MASK_COLS_PAIRS = [f"mask_{ch}" for ch in ALL_CHANNELS]
