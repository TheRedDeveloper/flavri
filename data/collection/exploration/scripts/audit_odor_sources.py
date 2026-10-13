import json
import pandas as pd
from rdkit import Chem

def inspect_dropped():
    df4986 = pd.read_csv('C:/Users/RedDev/clones/aromma-mcp/aromma/data/1_make_gslf/curated_GS_LF_merged_4986.csv')
    df4814 = pd.read_csv('C:/Users/RedDev/clones/aromma-mcp/aromma/data/1_make_gslf/gs-lf_combined.csv')

    s4986 = set(df4986['nonStereoSMILES'])
    s4814 = set(df4814['IsomericSMILES'])

    dropped_smiles = s4986 - s4814
    print(f"Total dropped between 4986 and 4814: {len(dropped_smiles)}")
    dropped_df = df4986[df4986['nonStereoSMILES'].isin(dropped_smiles)]
    
    print("\nBreakdown of dropped molecules:")
    # Check why they were dropped:
    # 1. Undesirable elements
    undesirable_elements = ["He", "Na", "Mg", "Al", "Si", "K", "Ca", "Ti", "V", "Cr", "Fe", "Co", "Cu", "Zn", "Bi"]
    metals = []
    charges = []
    multi = []
    high_mw = []
    low_mw = []
    non_carbon = []
    other = []

    for _, row in dropped_df.iterrows():
        s = row['nonStereoSMILES']
        desc = row['descriptors']
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            other.append((s, desc, "unparseable"))
            continue
        
        has_metal = any(mol.HasSubstructMatch(Chem.MolFromSmarts(f'[{el}]')) for el in undesirable_elements)
        if has_metal:
            metals.append((s, desc))
            continue
        if Chem.GetFormalCharge(mol) != 0:
            charges.append((s, desc))
            continue
        if '.' in s:
            multi.append((s, desc))
            continue
        from rdkit.Chem.Descriptors import ExactMolWt
        mw = ExactMolWt(mol)
        if mw < 20:
            low_mw.append((s, desc, mw))
            continue
        if mw > 600:
            high_mw.append((s, desc, mw))
            continue
        if 'C' not in s and 'c' not in s:
            non_carbon.append((s, desc))
            continue
        other.append((s, desc, "passed_all_filters"))

    print(f"  Metals/Inorganics: {len(metals)}")
    for s, d in metals[:10]:
        print(f"    {s} -> {d}")
    print(f"  Charges: {len(charges)}")
    for s, d in charges[:10]:
        print(f"    {s} -> {d}")
    print(f"  Multimolecules (mixtures/salts): {len(multi)}")
    for s, d in multi[:10]:
        print(f"    {s} -> {d}")
    print(f"  High MW (>600): {len(high_mw)}")
    for s, d, mw in high_mw[:5]:
        print(f"    {s} (MW={mw:.1f}) -> {d}")
    print(f"  Low MW (<20): {len(low_mw)}")
    for s, d, mw in low_mw:
        print(f"    {s} (MW={mw:.1f}) -> {d}")
    print(f"  Non-carbon: {len(non_carbon)}")
    for s, d in non_carbon:
        print(f"    {s} -> {d}")
    print(f"  Other: {len(other)}")
    for item in other:
        print(f"    {item}")

if __name__ == '__main__':
    inspect_dropped()
