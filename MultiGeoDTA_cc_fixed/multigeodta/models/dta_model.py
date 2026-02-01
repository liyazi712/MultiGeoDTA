"""
Drug-Target Affinity (DTA) Prediction Model

The main model architecture for MultiGeoDTA that integrates:
- Protein 3D structure encoding (GVP)
- Drug molecule 3D structure encoding (GVP)
- Protein sequence encoding (Mamba2/LSTM)
- SMILES sequence encoding (Mamba2/LSTM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from multigeodta.models.encoders import (
    ProtGVPModel, DrugGVPModel, SeqEncoder, SmileEncoder
)


class DTAModel(nn.Module):
    """
    Multi-modal Drug-Target Affinity Prediction Model.

    This model combines multiple modalities for binding affinity prediction:
    1. Protein 3D structure via GVP encoder
    2. Drug 3D structure via GVP encoder
    3. Protein sequence via Mamba2/LSTM encoder
    4. SMILES string via Mamba2/LSTM encoder

    The features from all modalities are fused through an MLP to predict
    binding affinity.

    Args:
        drug_node_in_dim: Drug node input dimensions (scalar, vector)
        drug_node_h_dims: Drug node hidden dimensions
        drug_edge_in_dim: Drug edge input dimensions
        drug_edge_h_dims: Drug edge hidden dimensions
        prot_node_in_dim: Protein node input dimensions
        prot_node_h_dims: Protein node hidden dimensions
        prot_edge_in_dim: Protein edge input dimensions
        prot_edge_h_dims: Protein edge hidden dimensions
        mlp_dims: Dimensions of the prediction MLP
        mlp_dropout: Dropout rate in MLP
        seq_embedding_dim: Dimension for sequence embeddings

    Example:
        >>> model = DTAModel()
        >>> pred, prot_feat, seq_feat, drug_feat, smile_feat = model(
        ...     drug_graph, protein_graph, protein_seq, pocket_seq, smile_seq
        ... )
    """

    def __init__(self,
                 # Drug GVP parameters
                 drug_node_in_dim: Tuple[int, int] = (86, 1),
                 drug_node_h_dims: Tuple[int, int] = (128, 64),
                 drug_edge_in_dim: Tuple[int, int] = (24, 3),
                 drug_edge_h_dims: Tuple[int, int] = (32, 1),
                 # Protein GVP parameters
                 prot_node_in_dim: Tuple[int, int] = (6, 3),
                 prot_node_h_dims: Tuple[int, int] = (128, 64),
                 prot_edge_in_dim: Tuple[int, int] = (32, 1),
                 prot_edge_h_dims: Tuple[int, int] = (32, 1),
                 # MLP parameters
                 mlp_dims: List[int] = [1024, 512],
                 mlp_dropout: float = 0.25,
                 # Sequence encoder parameters
                 seq_embedding_dim: int = 256):
        super(DTAModel, self).__init__()

        # Drug 3D structure encoder
        self.drug_GVP = DrugGVPModel(
            node_in_dim=drug_node_in_dim,
            node_h_dim=drug_node_h_dims,
            edge_in_dim=drug_edge_in_dim,
            edge_h_dim=drug_edge_h_dims,
            num_layers=1
        )

        # Protein 3D structure encoder
        self.prot_GVP = ProtGVPModel(
            node_in_dim=prot_node_in_dim,
            node_h_dim=prot_node_h_dims,
            edge_in_dim=prot_edge_in_dim,
            edge_h_dim=prot_edge_h_dims,
            num_layers=3
        )

        # Sequence encoders
        self.seq_encoder = SeqEncoder(embedding_dim=seq_embedding_dim)
        self.smile_encoder = SmileEncoder(embedding_dim=seq_embedding_dim)

        # Calculate hidden dimension
        hidden_dim = prot_node_h_dims[0]

        # Prediction MLP: combines all 4 modalities
        # Each modality contributes hidden_dim features (128)
        # Total: 128 * 4 = 512
        self.mlp = self._build_mlp(
            [hidden_dim * 4] + mlp_dims + [1],
            dropout=mlp_dropout,
            batchnorm=False,
            no_last_dropout=True,
            no_last_activation=True
        )

    def _build_mlp(self, hidden_sizes: List[int],
                   dropout: float = 0,
                   batchnorm: bool = False,
                   no_last_dropout: bool = True,
                   no_last_activation: bool = True) -> nn.Sequential:
        """
        Build MLP layers.

        Args:
            hidden_sizes: List of layer dimensions
            dropout: Dropout probability
            batchnorm: Whether to use batch normalization
            no_last_dropout: Skip dropout on last layer
            no_last_activation: Skip activation on last layer

        Returns:
            Sequential module containing MLP layers
        """
        act_fn = nn.ReLU()
        layers = []

        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))

            # Add activation (except possibly last layer)
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)

            # Add dropout (except possibly last layer)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))

            # Add batch norm (except last layer)
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))

        return nn.Sequential(*layers)

    def forward(self, xd, xp, protein_seq: torch.Tensor,
                pocket_seq: torch.Tensor, smile_seq: torch.Tensor) -> Tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for binding affinity prediction.

        Args:
            xd: Drug graph (PyG Data object)
            xp: Protein graph (PyG Data object)
            protein_seq: Protein sequence tensor (batch, seq_len)
            pocket_seq: Pocket sequence tensor (batch, seq_len)
            smile_seq: SMILES sequence tensor (batch, seq_len)

        Returns:
            Tuple of:
                - predictions: Binding affinity predictions (batch, 1)
                - protein_feats: Protein structure features (batch, hidden_dim)
                - seq_feats: Protein sequence features (batch, hidden_dim)
                - compound_feats: Drug structure features (batch, hidden_dim)
                - smile_feats: SMILES features (batch, hidden_dim)
        """
        # Encode protein 3D structure: (batch, max_nodes, hidden_dim) -> (batch, hidden_dim)
        protein_feats = torch.mean(self.prot_GVP(xp), dim=1)

        # Encode protein sequence: (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
        seq_feats = torch.mean(self.seq_encoder(protein_seq, pocket_seq), dim=1)

        # Encode drug 3D structure: (batch, max_atoms, hidden_dim) -> (batch, hidden_dim)
        compound_feats = torch.mean(self.drug_GVP(xd), dim=1)

        # Encode SMILES: (batch, smi_len, hidden_dim) -> (batch, hidden_dim)
        smile_feats = torch.mean(self.smile_encoder(smile_seq), dim=1)

        # Combine all features
        combined_feats = torch.cat([protein_feats, seq_feats, compound_feats, smile_feats], dim=-1)

        # Predict binding affinity
        predictions = self.mlp(combined_feats)

        return predictions, protein_feats, seq_feats, compound_feats, smile_feats

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self, encoder_name: str):
        """
        Freeze parameters of a specific encoder.

        Args:
            encoder_name: One of 'drug', 'protein', 'seq', 'smile'
        """
        encoder_map = {
            'drug': self.drug_GVP,
            'protein': self.prot_GVP,
            'seq': self.seq_encoder,
            'smile': self.smile_encoder
        }

        if encoder_name in encoder_map:
            for param in encoder_map[encoder_name].parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

    def unfreeze_encoder(self, encoder_name: str):
        """Unfreeze parameters of a specific encoder."""
        encoder_map = {
            'drug': self.drug_GVP,
            'protein': self.prot_GVP,
            'seq': self.seq_encoder,
            'smile': self.smile_encoder
        }

        if encoder_name in encoder_map:
            for param in encoder_map[encoder_name].parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
