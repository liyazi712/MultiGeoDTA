"""
Protein Graph Construction Module

Converts protein structures (PDB format) into PyTorch Geometric graphs
for use with GVP-based neural networks.

Adapted from:
- https://github.com/jingraham/neurips19-graph-protein-design
- https://github.com/drorlab/gvp-pytorch
"""

import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric
from scipy.spatial import KDTree
from typing import Dict, Tuple, Optional

from multigeodta.data.constants import LETTER_TO_NUM


def pdb_to_graphs(prot_data: Dict, params: Dict) -> Dict:
    """
    Convert protein data dictionaries to PyTorch Geometric graphs.

    Args:
        prot_data: Dictionary mapping protein IDs to structure data
        params: Parameters for graph construction (num_pos_emb, num_rbf, contact_cutoff)

    Returns:
        Dictionary mapping protein IDs to PyG Data objects
    """
    graphs = {}
    for key, struct in tqdm(prot_data.items(), desc='Building protein graphs'):
        graphs[key] = featurize_protein_graph(struct, name=key, **params)
    return graphs


def featurize_protein_graph(
    protein: Dict,
    name: Optional[str] = None,
    num_pos_emb: int = 16,
    num_rbf: int = 16,
    contact_cutoff: float = 8.0,
) -> torch_geometric.data.Data:
    """
    Convert a protein structure to a PyTorch Geometric graph.

    The graph contains:
    - Node features: Dihedral angles (scalar) and backbone orientations (vector)
    - Edge features: RBF of distances (scalar) and direction vectors (vector)

    Args:
        protein: Dictionary with 'coords' and 'seq' keys
        name: Protein identifier
        num_pos_emb: Number of positional embedding dimensions
        num_rbf: Number of radial basis function kernels
        contact_cutoff: Distance threshold for edge construction (Angstroms)

    Returns:
        PyTorch Geometric Data object
    """
    with torch.no_grad():
        # Parse coordinates: (N_residues, 4, 3) for N, CA, C, O atoms
        coords = torch.as_tensor(protein['coords'], dtype=torch.float32)
        seq = torch.as_tensor([LETTER_TO_NUM[a] for a in protein['seq']], dtype=torch.long)
        seq_emb = torch.load(protein['embed']) if 'embed' in protein else None

        # Create mask for valid coordinates
        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf

        # Extract C-alpha coordinates as node positions
        X_ca = coords[:, 1]  # CA atoms
        ca_mask = torch.isfinite(X_ca.sum(dim=1))
        ca_mask = ca_mask.float()
        ca_mask_2D = torch.unsqueeze(ca_mask, 0) * torch.unsqueeze(ca_mask, 1)

        # Compute pairwise distances
        dX_ca = torch.unsqueeze(X_ca, 0) - torch.unsqueeze(X_ca, 1)
        D_ca = ca_mask_2D * torch.sqrt(torch.sum(dX_ca**2, 2) + 1e-6)

        # Build edges based on distance cutoff
        edge_index = torch.nonzero((D_ca < contact_cutoff) & (ca_mask_2D == 1))

        # Add KNN edges for better connectivity
        knn_matrix = np.zeros_like(D_ca)
        tree = KDTree(D_ca)
        k = 3
        for i in range(D_ca.shape[0]):
            distances, indices = tree.query(D_ca[i], k=k + 1)
            knn_matrix[i, indices[1:]] = 1  # Exclude self

        knn_edge_index = torch.nonzero(torch.tensor(knn_matrix))
        combined_edge_index = torch.cat((edge_index, knn_edge_index), dim=0)
        edge_index = torch.unique(combined_edge_index, dim=0)
        edge_index = edge_index.t().contiguous()

        # Compute features
        pos_embeddings = _positional_embeddings(edge_index, num_embeddings=num_pos_emb)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf)
        dihedrals = _dihedrals(coords)
        orientations = _orientations(X_ca)
        sidechains = _sidechains(coords)

        # Assemble node and edge features
        node_s = dihedrals  # Scalar: (N, 6)
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)  # Vector: (N, 3, 3)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)  # Scalar: (E, num_rbf + num_pos_emb)
        edge_v = _normalize(E_vectors).unsqueeze(-2)  # Vector: (E, 1, 3)

        # Handle NaN values
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

    data = torch_geometric.data.Data(
        x=X_ca, seq=seq, name=name, coords=coords,
        node_s=node_s, node_v=node_v,
        edge_s=edge_s, edge_v=edge_v,
        edge_index=edge_index, mask=mask,
        seq_emb=seq_emb
    )
    return data


def _dihedrals(X: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute dihedral angle features for protein backbone.

    Args:
        X: Backbone coordinates (N, 4, 3) for N, CA, C, O

    Returns:
        Dihedral features (N, 6) with cos and sin of angles
    """
    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]

    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Compute plane normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Compute dihedral angles
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])

    # Convert to periodic representation (cos, sin)
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index: torch.Tensor, num_embeddings: int = 16) -> torch.Tensor:
    """
    Compute positional embeddings based on sequence distance.

    Args:
        edge_index: Edge indices (2, E)
        num_embeddings: Number of embedding dimensions

    Returns:
        Positional embeddings (E, num_embeddings)
    """
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32) *
        -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def _orientations(X: torch.Tensor) -> torch.Tensor:
    """
    Compute backbone orientation features.

    Args:
        X: C-alpha coordinates (N, 3)

    Returns:
        Forward and backward direction vectors (N, 2, 3)
    """
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X: torch.Tensor) -> torch.Tensor:
    """
    Compute sidechain direction vectors.

    Args:
        X: Backbone coordinates (N, 4, 3)

    Returns:
        Sidechain direction vectors (N, 3)
    """
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize tensor along specified dimension without NaN values."""
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _rbf(D: torch.Tensor, D_min: float = 0., D_max: float = 20.,
         D_count: int = 16, device: str = 'cpu') -> torch.Tensor:
    """
    Compute radial basis function embedding of distances.

    Args:
        D: Distance tensor
        D_min: Minimum distance
        D_max: Maximum distance
        D_count: Number of RBF kernels

    Returns:
        RBF embeddings (..., D_count)
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
