"""
Molecule Graph Construction Module

Converts molecular structures (SDF/MOL format) into PyTorch Geometric graphs
for use with GVP-based neural networks.
"""

import numpy as np
import pandas as pd
import torch
import torch_geometric
from rdkit import Chem
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from multigeodta.data.constants import (
    ATOM_VOCAB, ELECTRONEGATIVITY, COVALENT_RADIUS,
    VDW_RADIUS, VALENCE_ELECTRONS
)
from multigeodta.data.protein_graph import _rbf, _normalize


def onehot_encoder(a: List, alphabet: List, default=None,
                   drop_first: bool = False) -> np.ndarray:
    """
    One-hot encode categorical features.

    Args:
        a: List of categorical values
        alphabet: Valid categories
        default: Default value for out-of-vocabulary items
        drop_first: Whether to drop first column (for binary features)

    Returns:
        One-hot encoded array (N, len(alphabet))
    """
    alphabet_set = set(alphabet)
    a = [x if x in alphabet_set else default for x in a]
    a = pd.Categorical(a, categories=alphabet)
    onehot = pd.get_dummies(pd.Series(a), columns=alphabet, drop_first=drop_first)
    return onehot.values


def get_edge_index_from_mol(mol: Chem.Mol) -> torch.Tensor:
    """
    Extract edge indices from RDKit molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Edge index tensor (2, num_edges)
    """
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def build_atom_features(mol: Chem.Mol) -> np.ndarray:
    """
    Build comprehensive atom features for a molecule.

    Features include:
    - Atom symbol (one-hot)
    - Degree
    - Total hydrogens
    - Implicit valence
    - Aromaticity
    - Formal charge
    - Hybridization
    - Ring membership
    - Chirality
    - H-bond donor/acceptor
    - Atomic mass (normalized)
    - Electronegativity (normalized)
    - Covalent radius (normalized)
    - Van der Waals radius (normalized)
    - Partial charge
    - Valence electrons (normalized)

    Args:
        mol: RDKit Mol object

    Returns:
        Atom feature array (num_atoms, num_features)
    """
    if isinstance(mol, Chem.Mol):
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        except:
            pass

    # Feature specifications: (alphabet, default)
    feature_specs = {
        'GetSymbol': (ATOM_VOCAB, 'unk'),
        'GetDegree': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetTotalNumHs': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetImplicitValence': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetIsAromatic': ([0, 1], 1),
        'GetFormalCharge': ([-2, -1, 0, 1, 2], 0),
        'GetHybridization': ([
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2
        ], Chem.HybridizationType.SP3),
        'IsInRing': ([0, 1], 1),
        'IsChiral': ([0, 1], 0),
        'IsHDonor': ([0, 1], 0),
        'IsHAcceptor': ([0, 1], 0),
    }

    atom_feature = None

    # Process discrete features
    for attr in ['GetSymbol', 'GetDegree', 'GetTotalNumHs', 'GetImplicitValence',
                 'GetIsAromatic', 'GetFormalCharge', 'GetHybridization', 'IsInRing',
                 'IsChiral', 'IsHDonor', 'IsHAcceptor']:
        if attr == 'IsChiral':
            feature = []
            for atom in mol.GetAtoms():
                if hasattr(atom, 'IsChiral'):
                    feature.append(1 if atom.IsChiral() else 0)
                else:
                    feature.append(0)
        elif attr == 'IsHDonor':
            feature = [1 if (atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0) else 0
                      for atom in mol.GetAtoms()]
        elif attr == 'IsHAcceptor':
            feature = [1 if atom.GetSymbol() in ['N', 'O'] else 0
                      for atom in mol.GetAtoms()]
        else:
            feature = [getattr(atom, attr)() for atom in mol.GetAtoms()]

        feature = onehot_encoder(
            feature,
            alphabet=feature_specs[attr][0],
            default=feature_specs[attr][1],
            drop_first=(attr in ['GetIsAromatic', 'IsInRing', 'IsChiral', 'IsHDonor', 'IsHAcceptor'])
        )

        atom_feature = feature if atom_feature is None else np.concatenate((atom_feature, feature), axis=1)

    # Process continuous features
    continuous_features = []

    # Atomic mass (normalized to [0, 1] for range [0, 200])
    mass = [(atom.GetMass() - 0) / 200 for atom in mol.GetAtoms()]
    continuous_features.append(np.array(mass).reshape(-1, 1))

    # Electronegativity
    en = [ELECTRONEGATIVITY.get(atom.GetSymbol(), 2.5) for atom in mol.GetAtoms()]
    en = [(x - 0.5) / 3.5 for x in en]  # Normalize to [0, 1]
    continuous_features.append(np.array(en).reshape(-1, 1))

    # Covalent radius
    cr = [COVALENT_RADIUS.get(atom.GetSymbol(), 1.0) for atom in mol.GetAtoms()]
    cr = [(x - 0.3) / 1.7 for x in cr]
    continuous_features.append(np.array(cr).reshape(-1, 1))

    # Van der Waals radius
    vdw = [VDW_RADIUS.get(atom.GetSymbol(), 1.5) for atom in mol.GetAtoms()]
    vdw = [(x - 1.0) / 2.0 for x in vdw]
    continuous_features.append(np.array(vdw).reshape(-1, 1))

    # Partial charge (if available)
    pc = [atom.GetDoubleProp('PartialCharge') if atom.HasProp('PartialCharge') else 0.0
          for atom in mol.GetAtoms()]
    pc = [(x + 1.0) / 2.0 for x in pc]  # Normalize from [-1, 1] to [0, 1]
    continuous_features.append(np.array(pc).reshape(-1, 1))

    # Valence electrons
    ve = [VALENCE_ELECTRONS.get(atom.GetSymbol(), 4) for atom in mol.GetAtoms()]
    ve = [x / 18 for x in ve]
    continuous_features.append(np.array(ve).reshape(-1, 1))

    # Combine all features
    continuous_feature = np.concatenate(continuous_features, axis=1)
    atom_feature = np.concatenate((atom_feature, continuous_feature), axis=1)

    return atom_feature.astype(np.float32)


def build_edge_features(mol: Chem.Mol, coords: torch.Tensor,
                       edge_index: torch.Tensor, D_max: float = 4.5,
                       num_rbf: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge features for molecular graph.

    Features include:
    - RBF of distances (scalar)
    - Bond type (one-hot)
    - Conjugation
    - Ring membership
    - Bond length deviation
    - Electronegativity difference
    - Direction vectors (vector)

    Args:
        mol: RDKit Mol object
        coords: Atom coordinates (N, 3)
        edge_index: Edge indices (2, E)
        D_max: Maximum distance for RBF
        num_rbf: Number of RBF kernels

    Returns:
        Tuple of (edge_scalar_features, edge_vector_features)
    """
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    distances = E_vectors.norm(dim=-1)
    rbf = _rbf(distances, D_max=D_max, D_count=num_rbf)

    edge_s_features = [rbf]

    # Bond length lookup
    bond_length_map = {
        (Chem.BondType.SINGLE, 'C', 'C'): 1.54,
        (Chem.BondType.SINGLE, 'C', 'N'): 1.47,
        (Chem.BondType.SINGLE, 'C', 'O'): 1.43,
        (Chem.BondType.DOUBLE, 'C', 'C'): 1.34,
        (Chem.BondType.DOUBLE, 'C', 'O'): 1.20,
        (Chem.BondType.TRIPLE, 'C', 'C'): 1.20,
    }

    bond_types = []
    is_conjugated = []
    is_in_ring = []
    distance_deviations = []
    electronegativity_diffs = []

    for i in range(edge_index.shape[1]):
        idx0, idx1 = int(edge_index[0, i]), int(edge_index[1, i])
        atom0, atom1 = mol.GetAtomWithIdx(idx0), mol.GetAtomWithIdx(idx1)
        bond = mol.GetBondBetweenAtoms(idx0, idx1)

        if bond is not None:
            bond_type = bond.GetBondType()
            bond_types.append([
                bond_type == Chem.BondType.SINGLE,
                bond_type == Chem.BondType.DOUBLE,
                bond_type == Chem.BondType.TRIPLE,
                bond_type == Chem.BondType.AROMATIC
            ])
            is_conjugated.append(bond.GetIsConjugated())
            is_in_ring.append(bond.IsInRing())

            # Bond length deviation
            key = (bond.GetBondType(), atom0.GetSymbol(), atom1.GetSymbol())
            key_reverse = (bond.GetBondType(), atom1.GetSymbol(), atom0.GetSymbol())
            std_length = bond_length_map.get(key, bond_length_map.get(key_reverse, 1.5))
            actual_length = distances[i].item()
            distance_deviations.append((actual_length - std_length) / std_length)
        else:
            bond_types.append([1, 0, 0, 0])
            is_conjugated.append(False)
            is_in_ring.append(False)
            distance_deviations.append(0.0)

        # Electronegativity difference
        en0 = ELECTRONEGATIVITY.get(atom0.GetSymbol(), 2.5)
        en1 = ELECTRONEGATIVITY.get(atom1.GetSymbol(), 2.5)
        electronegativity_diffs.append(abs(en0 - en1))

    # Convert to tensors
    bond_types = torch.tensor(bond_types, dtype=torch.float32)
    is_conjugated = torch.tensor(is_conjugated, dtype=torch.float32).unsqueeze(-1)
    is_in_ring = torch.tensor(is_in_ring, dtype=torch.float32).unsqueeze(-1)
    distance_deviations = torch.tensor(distance_deviations, dtype=torch.float32).unsqueeze(-1)
    electronegativity_diffs = torch.tensor(electronegativity_diffs, dtype=torch.float32).unsqueeze(-1)

    # Combine scalar features
    edge_s_features.extend([bond_types, is_conjugated, is_in_ring,
                           distance_deviations, electronegativity_diffs])
    edge_s = torch.cat(edge_s_features, dim=-1)

    # Build vector features
    edge_v_features = []
    edge_v_features.append(_normalize(E_vectors).unsqueeze(-2))
    edge_v_features.append(E_vectors.unsqueeze(-2))

    # Electronegativity gradient vector
    en_diff_vectors = []
    for i in range(edge_index.shape[1]):
        idx0, idx1 = int(edge_index[0, i]), int(edge_index[1, i])
        atom0, atom1 = mol.GetAtomWithIdx(idx0), mol.GetAtomWithIdx(idx1)
        en0 = ELECTRONEGATIVITY.get(atom0.GetSymbol(), 2.5)
        en1 = ELECTRONEGATIVITY.get(atom1.GetSymbol(), 2.5)
        en_diff_vector = E_vectors[i] * (en1 - en0) / (distances[i] + 1e-8)
        en_diff_vectors.append(en_diff_vector)
    edge_v_features.append(torch.stack(en_diff_vectors).unsqueeze(-2))

    edge_v = torch.cat(edge_v_features, dim=-2)

    # Handle NaN values
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


def sdf_to_graphs(data_list: Dict) -> Dict:
    """
    Convert molecule data to PyTorch Geometric graphs.

    Args:
        data_list: Dictionary mapping molecule IDs to structure data

    Returns:
        Dictionary mapping molecule IDs to PyG Data objects
    """
    graphs = {}
    for key, value in tqdm(data_list.items(), desc='Building molecule graphs'):
        graphs[key] = featurize_drug(
            mol=value['mol'],
            coords=value['coords'],
            atom_type=value['atom_types'],
            name=key
        )
    return graphs


def featurize_drug(mol: Chem.Mol, coords: np.ndarray, atom_type,
                   name: Optional[str] = None, edge_cutoff: float = 4.5,
                   num_rbf: int = 16) -> torch_geometric.data.Data:
    """
    Convert a molecule to a PyTorch Geometric graph.

    Args:
        mol: RDKit Mol object
        coords: Atom coordinates array (N, 3)
        atom_type: Atom types
        name: Molecule identifier
        edge_cutoff: Distance cutoff for edges
        num_rbf: Number of RBF kernels

    Returns:
        PyTorch Geometric Data object
    """
    with torch.no_grad():
        coords = torch.as_tensor(coords, dtype=torch.float32)
        atom_feature = build_atom_features(mol)
        atom_feature = torch.as_tensor(atom_feature, dtype=torch.float32)
        edge_index = get_edge_index_from_mol(mol)

    node_s = atom_feature
    node_v = coords.unsqueeze(1)
    edge_s, edge_v = build_edge_features(mol, coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf)

    data = torch_geometric.data.Data(
        x=atom_type,
        edge_index=edge_index,
        name=name,
        node_v=node_v,
        node_s=node_s,
        edge_v=edge_v,
        edge_s=edge_s
    )
    return data
