"""
flavri.featurizer
Molecular graph featurizer for small molecules and oligopeptides.
Produces atom features (d_atom = 133) and directed bond features (d_bond = 14).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)


# ---------------------------------------------------------------------------
# Feature Dimensions & Choice Dictionaries
# ---------------------------------------------------------------------------

MAX_ATOMIC_NUM = 100  # elements 1 to 100 + 1 unk = 101 dims
DEGREE_CHOICES = [0, 1, 2, 3, 4, 5]  # 6 + 1 unk = 7 dims
FORMAL_CHARGE_CHOICES = [-2, -1, 0, 1, 2]  # 5 + 1 unk = 6 dims
CHIRAL_TAG_CHOICES = [0, 1, 2, 3]  # 4 + 1 unk = 5 dims
NUM_HS_CHOICES = [0, 1, 2, 3, 4]  # 5 + 1 unk = 6 dims
HYBRIDIZATION_CHOICES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]  # 5 + 1 unk = 6 dims

ATOM_FDIM = 101 + 7 + 6 + 5 + 6 + 6 + 1 + 1  # 133
BOND_FDIM = 4 + 1 + 1 + 7 + 1                 # 14


def one_hot_unk(val: Any, choices: List[Any]) -> List[float]:
    """One-hot encodes value with trailing slot for unknown / out-of-bounds."""
    encoding = [0.0] * (len(choices) + 1)
    if val in choices:
        encoding[choices.index(val)] = 1.0
    else:
        encoding[-1] = 1.0
    return encoding


def get_atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    """
    Computes 133-dimensional atom feature vector.
    - Atomic number (one-hot 1..100, unk): 101
    - Total degree (0..5, unk): 7
    - Formal charge (-2..2, unk): 6
    - Chiral tag (0..3, unk): 5
    - Total Hs (0..4, unk): 6
    - Hybridization (SP..SP3D2, unk): 6
    - Aromaticity (binary): 1
    - Scaled mass: 1
    Total = 133
    """
    atomic_num = atom.GetAtomicNum()
    at_choices = list(range(1, MAX_ATOMIC_NUM + 1))
    
    feats = (
        one_hot_unk(atomic_num, at_choices)
        + one_hot_unk(atom.GetTotalDegree(), DEGREE_CHOICES)
        + one_hot_unk(atom.GetFormalCharge(), FORMAL_CHARGE_CHOICES)
        + one_hot_unk(int(atom.GetChiralTag()), CHIRAL_TAG_CHOICES)
        + one_hot_unk(int(atom.GetTotalNumHs()), NUM_HS_CHOICES)
        + one_hot_unk(atom.GetHybridization(), HYBRIDIZATION_CHOICES)
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [atom.GetMass() * 0.01]
    )
    assert len(feats) == ATOM_FDIM, f"Atom feature dim mismatch: expected {ATOM_FDIM}, got {len(feats)}"
    return feats


def get_bond_features(bond: Optional[Chem.rdchem.Bond]) -> List[float]:
    """
    Computes 14-dimensional bond feature vector.
    - Bond type: Single, Double, Triple, Aromatic (4)
    - Conjugated (1)
    - In Ring (1)
    - Stereo: None, Any, Z, E, Cis, Trans + unk (7)
    - Direction (1)
    Total = 14
    """
    if bond is None:
        return [0.0] * BOND_FDIM

    bt = bond.GetBondType()
    stereo_choices = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    
    feats = [
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]
    feats += one_hot_unk(bond.GetStereo(), stereo_choices)
    feats += [1.0 if bond.GetBondDir() != Chem.rdchem.BondDir.NONE else 0.0]
    
    assert len(feats) == BOND_FDIM, f"Bond feature dim mismatch: expected {BOND_FDIM}, got {len(feats)}"
    return feats


# ---------------------------------------------------------------------------
# Directed Graph Data Structure
# ---------------------------------------------------------------------------

class MolGraph:
    """
    Directed bond graph representation for a single molecule.
    """
    def __init__(self, smiles: str):
        self.smiles = smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Fallback for sanitization issues
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)

        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")

        self.n_atoms = mol.GetNumAtoms()
        # Fallback for single atom without bonds
        if self.n_atoms == 0:
            raise ValueError(f"SMILES has 0 atoms: {smiles}")

        # Morgan / ECFP4 fingerprint (512 bits)
        self.fp: np.ndarray = _MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)

        # Atom features: (n_atoms, ATOM_FDIM)
        self.f_atoms: List[List[float]] = [
            get_atom_features(atom) for atom in mol.GetAtoms()
        ]

        # Directed bonds: 1-indexed (0 reserved for padding/null)
        # Every undirected bond generates two directed bonds: (u -> v) and (v -> u)
        self.f_bonds: List[List[float]] = [[0.0] * BOND_FDIM]  # index 0: padding
        self.u_atoms: List[int] = [0]  # source atom for bond
        self.v_atoms: List[int] = [0]  # target atom for bond
        self.b2rev: List[int] = [0]    # reverse bond index

        # a2b: mapping each atom to incoming directed bond indices (1-indexed)
        self.a2b: List[List[int]] = [[] for _ in range(self.n_atoms)]
        # b2b: mapping each directed bond to list of incoming bonds to u(b) except rev(b)
        self.b2b: List[List[int]] = [[]]

        n_bonds = 0
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            fbond = get_bond_features(bond)

            # Directed bond 1: u -> v
            n_bonds += 1
            b1 = n_bonds
            self.f_bonds.append(fbond)
            self.u_atoms.append(u)
            self.v_atoms.append(v)
            self.a2b[v].append(b1)

            # Directed bond 2: v -> u
            n_bonds += 1
            b2 = n_bonds
            self.f_bonds.append(fbond)
            self.u_atoms.append(v)
            self.v_atoms.append(u)
            self.a2b[u].append(b2)

            self.b2rev.append(b2)
            self.b2rev.append(b1)

        self.n_bonds = n_bonds

        # Precompute b2b: for bond b=(u->v), incoming bonds to u excluding reverse bond
        for b in range(1, self.n_bonds + 1):
            u = self.u_atoms[b]
            rev = self.b2rev[b]
            incoming_to_u = [k for k in self.a2b[u] if k != rev]
            self.b2b.append(incoming_to_u)


class BatchMolGraph:
    """
    Batched directed molecular graphs with dense index tensors for GPU execution.
    """
    def __init__(self, mol_graphs: List[MolGraph], device: torch.device = torch.device("cpu")):
        self.smiles_batch = [g.smiles for g in mol_graphs]
        self.batch_size = len(mol_graphs)
        self.device = device

        f_atoms_all = []
        f_bonds_all = [[0.0] * BOND_FDIM]
        
        # Atom to graph mapping for pooling
        a2g_list = []
        # Bond to target atom (1-indexed for bonds)
        b2a_list = [0]
        # b2u_list: source atom
        b2u_list = [0]

        total_atoms = 0
        total_bonds = 0

        # For vectorized message passing:
        # Max incoming degree per bond across batch
        max_incoming_bonds = 1
        for g in mol_graphs:
            for incoming in g.b2b[1:]:
                if len(incoming) > max_incoming_bonds:
                    max_incoming_bonds = len(incoming)

        # Max incoming degree per atom across batch
        max_incoming_atoms = 1
        for g in mol_graphs:
            for incoming in g.a2b:
                if len(incoming) > max_incoming_atoms:
                    max_incoming_atoms = len(incoming)

        b2b_list = [[]]  # index 0
        a2b_list = []

        for g_idx, g in enumerate(mol_graphs):
            f_atoms_all.extend(g.f_atoms)
            for _ in range(g.n_atoms):
                a2g_list.append(g_idx)

            for b in range(1, g.n_bonds + 1):
                f_bonds_all.append(g.f_bonds[b])
                b2a_list.append(g.v_atoms[b] + total_atoms)
                b2u_list.append(g.u_atoms[b] + total_atoms)
                # Offset incoming bonds by total_bonds
                incoming = [inc + total_bonds for inc in g.b2b[b]]
                b2b_list.append(incoming)

            for a in range(g.n_atoms):
                incoming_to_a = [inc + total_bonds for inc in g.a2b[a]]
                a2b_list.append(incoming_to_a)

            total_atoms += g.n_atoms
            total_bonds += g.n_bonds

        self.total_atoms = total_atoms
        self.total_bonds = total_bonds

        # Construct padded tensor for b2b: (total_bonds + 1, max_incoming_bonds)
        b2b_tensor = torch.zeros((total_bonds + 1, max_incoming_bonds), dtype=torch.long)
        for b, incoming in enumerate(b2b_list):
            if incoming:
                b2b_tensor[b, :len(incoming)] = torch.tensor(incoming, dtype=torch.long)

        # Construct padded tensor for a2b: (total_atoms, max_incoming_atoms)
        a2b_tensor = torch.zeros((total_atoms, max_incoming_atoms), dtype=torch.long)
        for a, incoming in enumerate(a2b_list):
            if incoming:
                a2b_tensor[a, :len(incoming)] = torch.tensor(incoming, dtype=torch.long)

        self.f_atoms = torch.tensor(f_atoms_all, dtype=torch.float32, device=device)
        self.f_bonds = torch.tensor(f_bonds_all, dtype=torch.float32, device=device)
        self.a2g = torch.tensor(a2g_list, dtype=torch.long, device=device)
        self.b2a = torch.tensor(b2a_list, dtype=torch.long, device=device)
        self.b2u = torch.tensor(b2u_list, dtype=torch.long, device=device)
        self.b2b = b2b_tensor.to(device)
        self.a2b = a2b_tensor.to(device)
        self.fps = torch.tensor(np.stack([g.fp for g in mol_graphs]), dtype=torch.float32, device=device)


# In-memory graph cache to eliminate redundant RDKit featurization
_GRAPH_CACHE: Dict[str, MolGraph] = {}


def get_mol_graph(smiles: str) -> MolGraph:
    """Retrieves or builds cached MolGraph."""
    if smiles not in _GRAPH_CACHE:
        _GRAPH_CACHE[smiles] = MolGraph(smiles)
    return _GRAPH_CACHE[smiles]


def mol2graph(smiles_list: List[str], device: torch.device = torch.device("cpu")) -> BatchMolGraph:
    """Batches list of SMILES into a BatchMolGraph."""
    graphs = [get_mol_graph(s) for s in smiles_list]
    return BatchMolGraph(graphs, device=device)
