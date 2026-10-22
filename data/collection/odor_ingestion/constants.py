"""
constants.py
Canonical vocabulary and constants for FLAVRI_ODOR dataset construction.
"""

from pathlib import Path

# Base Paths - Self-contained within flavri repository (no dependency on aromma-mcp)
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_ODOR_DIR = BASE_DIR / "exploration" / "raw" / "odor"

RAW_GOODSCENTS = RAW_ODOR_DIR / "curated_goodcents.csv"
RAW_LEFFINGWELL = RAW_ODOR_DIR / "curated_leffingwell.csv"
RAW_MMD = RAW_ODOR_DIR / "mmd_alldesc_6711.csv"
RAW_OLFACTIONBASE_ODORLESS = RAW_ODOR_DIR / "olfactionbase_odorless.csv"
RAW_BP_DIR = RAW_ODOR_DIR / "bp"

OUTPUT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_SINGLES_CSV = OUTPUT_DIR / "flavri_odor_singles.csv"
OUTPUT_PAIRS_CSV = OUTPUT_DIR / "flavri_odor_pairs.csv"

# 137 Canonical GS-LF sensory descriptors (alphabetical, odorless removed)
CANONICAL_ODOR_DESCRIPTORS = [
    "alcoholic",
    "aldehydic",
    "alliaceous",
    "almond",
    "amber",
    "animal",
    "anisic",
    "apple",
    "apricot",
    "aromatic",
    "balsamic",
    "banana",
    "beefy",
    "bergamot",
    "berry",
    "bitter",
    "black currant",
    "brandy",
    "burnt",
    "buttery",
    "cabbage",
    "camphoreous",
    "caramellic",
    "cedar",
    "celery",
    "chamomile",
    "cheesy",
    "cherry",
    "chocolate",
    "cinnamon",
    "citrus",
    "clean",
    "clove",
    "cocoa",
    "coconut",
    "coffee",
    "cognac",
    "cooked",
    "cooling",
    "cortex",
    "coumarinic",
    "creamy",
    "cucumber",
    "dairy",
    "dry",
    "earthy",
    "ethereal",
    "fatty",
    "fermented",
    "fishy",
    "floral",
    "fresh",
    "fruit skin",
    "fruity",
    "garlic",
    "gassy",
    "geranium",
    "grape",
    "grapefruit",
    "grassy",
    "green",
    "hawthorn",
    "hay",
    "hazelnut",
    "herbal",
    "honey",
    "hyacinth",
    "jasmin",
    "juicy",
    "ketonic",
    "lactonic",
    "lavender",
    "leafy",
    "leathery",
    "lemon",
    "lily",
    "malty",
    "meaty",
    "medicinal",
    "melon",
    "metallic",
    "milky",
    "mint",
    "muguet",
    "mushroom",
    "musk",
    "musty",
    "natural",
    "nutty",
    "oily",
    "onion",
    "orange",
    "orangeflower",
    "orris",
    "ozone",
    "peach",
    "pear",
    "phenolic",
    "pine",
    "pineapple",
    "plum",
    "popcorn",
    "potato",
    "powdery",
    "pungent",
    "radish",
    "raspberry",
    "ripe",
    "roasted",
    "rose",
    "rummy",
    "sandalwood",
    "savory",
    "sharp",
    "smoky",
    "soapy",
    "solvent",
    "sour",
    "spicy",
    "strawberry",
    "sulfurous",
    "sweaty",
    "sweet",
    "tea",
    "terpenic",
    "tobacco",
    "tomato",
    "tropical",
    "vanilla",
    "vegetable",
    "vetiver",
    "violet",
    "warm",
    "waxy",
    "weedy",
    "winey",
    "woody",
]

# Chemical synonym mappings
SYNONYM_MAP = {
    "fungal": "mushroom",
    "licorice": "anisic",
    "tonka": "coumarinic",
    "mentholic": "mint",
    "acidic": "sour",
    "marine": "ozone",
}

# Rigorous mapping from MMD (Merged Molecular Dataset / Sajan 2026) descriptors
# to Canonical 137 GS-LF sensory descriptors (only clean equivalents and sublabels).
MMD_MORPHOLOGICAL_MAP = {
    # Morphological variants (noun -> canonical adjective)
    "alcohol": "alcoholic",
    "aldehyde": "aldehydic",
    "anise": "anisic",
    "balsam": "balsamic",
    "beef": "beefy",
    "butter": "buttery",
    "camphor": "camphoreous",
    "caramel": "caramellic",
    "cheese": "cheesy",
    "coumarin": "coumarinic",
    "cream": "creamy",
    "earth": "earthy",
    "ether": "ethereal",
    "fat": "fatty",
    "fish": "fishy",
    "grass": "grassy",
    "jasmine": "jasmin",
    "juice": "juicy",
    "leaf": "leafy",
    "malt": "malty",
    "meat": "meaty",
    "milk": "milky",
    "phenol": "phenolic",
    "rum": "rummy",
    "smoked": "smoky",
    "sulfur": "sulfurous",
    "sweat": "sweaty",
    "terpene": "terpenic",
    "wax": "waxy",
    "wine": "winey",
}

MMD_CLEAN_SUBLABELS_AND_SYNONYMS = {
    # Clean sublabels (taxonomic subordinates to canonical parent)
    "flower": "floral",
    "lilac": "floral",
    "gardenia": "floral",
    "mimosa": "floral",
    "narcissus": "floral",
    # Direct sensory synonyms
    "menthol": "mint",
    "acid": "sour",
    "marine": "ozone",
    "watery": "fresh",
    "neroli": "orangeflower",
}

# 16 Phantom / controversial MMD descriptors discarded to prevent label noise and blurring
MMD_DISCARDED_PHANTOMS = {
    "moss",         # Distinct oakmoss/lichens nuance; not generic earthy
    "bread",        # Fermented/baked crust note; not roasted pyrazine
    "chicken",      # Specific poultry nuance; avoid diluting meaty
    "broth",        # Liquid savory extract
    "plastic",      # Synthetic off-note
    "chemical",     # Non-specific artifact
    "jam",          # Cooked preserve; distinct from raw fruit
    "gourmand",     # Broad umbrella term
    "patchouli",    # Specific sesquiterpene camphoreous/woody; do not blur
    "horseradish",  # Glucosinolate tear-gas trigeminal
    "pepper",       # Piperine/caryophyllene distinct spice
    "peel",         # Bitter terpene rinds
    "rooty",        # Rhizome note
    "licorice",     # Glycyrrhizin / anethole
    "gasoline",     # Petroleum hydrocarbon
    "ester",        # Broad organic chemical class
}

# 8 Phantom labels discarded from blend pairs
DISCARD_PHANTOMS = {
    "mossy",
    "bready",
    "brown",
    "chemical",
    "estery",
    "fusel",
    "jammy",
    "thujonic",
}

