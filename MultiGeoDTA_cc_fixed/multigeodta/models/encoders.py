"""
Encoder Modules for MultiGeoDTA

Contains protein and drug encoder architectures:
- ProtGVPModel: Protein structure encoder using GVP
- DrugGVPModel: Drug molecule encoder using GVP
- SeqEncoder: Protein sequence encoder using Mamba2 State Space Model
- SmileEncoder: SMILES sequence encoder using Mamba2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from multigeodta.models.gvp import GVP, GVPConvLayer, LayerNorm

try:
    from mamba_ssm import Mamba2
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: mamba_ssm not installed. SeqEncoder will use LSTM fallback.")


class ProtGVPModel(nn.Module):
    """
    Protein structure encoder using Geometric Vector Perceptron.

    Encodes protein 3D structure into fixed-size representations by processing
    the protein graph with GVP convolution layers.

    Args:
        node_in_dim: Input node feature dimensions (scalar, vector)
        node_h_dim: Hidden node feature dimensions (scalar, vector)
        edge_in_dim: Input edge feature dimensions (scalar, vector)
        edge_h_dim: Hidden edge feature dimensions (scalar, vector)
        num_layers: Number of GVP convolution layers
        drop_rate: Dropout probability
    """

    def __init__(self,
                 node_in_dim: Tuple[int, int] = (6, 3),
                 node_h_dim: Tuple[int, int] = (128, 64),
                 edge_in_dim: Tuple[int, int] = (32, 1),
                 edge_h_dim: Tuple[int, int] = (32, 1),
                 num_layers: int = 3,
                 drop_rate: float = 0.1):
        super(ProtGVPModel, self).__init__()

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0))
        )

    def pyg_split(self, batched_data, feats: torch.Tensor) -> torch.Tensor:
        """
        Split batched features back into padded batch format.

        Args:
            batched_data: PyTorch Geometric batched data
            feats: Node features tensor

        Returns:
            Padded tensor of shape (batch_size, max_nodes, features)
        """
        device = feats.device
        batch_size = batched_data.ptr.size(0) - 1
        node_to_graph_idx = batched_data.batch
        num_nodes_per_graph = torch.bincount(node_to_graph_idx)
        max_num_nodes = int(num_nodes_per_graph.max())

        batch = torch.cat(
            [torch.full((1, x.type(torch.int)), y) for x, y in zip(num_nodes_per_graph, range(batch_size))],
            dim=1
        ).reshape(-1).type(torch.long).to(device)

        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes_per_graph.cumsum(dim=0)])
        idx = torch.arange(len(node_to_graph_idx), dtype=torch.long, device=device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def forward(self, xp) -> torch.Tensor:
        """
        Forward pass for protein encoding.

        Args:
            xp: Protein graph data with node_s, node_v, edge_s, edge_v, edge_index

        Returns:
            Protein representations of shape (batch_size, max_nodes, hidden_dim)
        """
        h_V = (xp.node_s, xp.node_v)
        h_E = (xp.edge_s, xp.edge_v)
        edge_index = xp.edge_index

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        out = self.W_out(h_V)
        out = self.pyg_split(xp, out)
        return out


class DrugGVPModel(nn.Module):
    """
    Drug molecule encoder using Geometric Vector Perceptron.

    Encodes molecular 3D structure into fixed-size representations.

    Args:
        node_in_dim: Input node feature dimensions (scalar, vector)
        node_h_dim: Hidden node feature dimensions (scalar, vector)
        edge_in_dim: Input edge feature dimensions (scalar, vector)
        edge_h_dim: Hidden edge feature dimensions (scalar, vector)
        num_layers: Number of GVP convolution layers
        drop_rate: Dropout probability
    """

    def __init__(self,
                 node_in_dim: Tuple[int, int] = (86, 1),
                 node_h_dim: Tuple[int, int] = (128, 64),
                 edge_in_dim: Tuple[int, int] = (24, 3),
                 edge_h_dim: Tuple[int, int] = (32, 1),
                 num_layers: int = 1,
                 drop_rate: float = 0.1):
        super(DrugGVPModel, self).__init__()

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0))
        )

    def pyg_split(self, batched_data, feats: torch.Tensor) -> torch.Tensor:
        """Split batched features into padded batch format."""
        device = feats.device
        batch_size = batched_data.ptr.size(0) - 1
        node_to_graph_idx = batched_data.batch
        num_nodes_per_graph = torch.bincount(node_to_graph_idx)
        max_num_nodes = int(num_nodes_per_graph.max())

        batch = torch.cat(
            [torch.full((1, x.type(torch.int)), y) for x, y in zip(num_nodes_per_graph, range(batch_size))],
            dim=1
        ).reshape(-1).type(torch.long).to(device)

        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes_per_graph.cumsum(dim=0)])
        idx = torch.arange(len(node_to_graph_idx), dtype=torch.long, device=device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def forward(self, xd) -> torch.Tensor:
        """
        Forward pass for drug encoding.

        Args:
            xd: Drug graph data with node_s, node_v, edge_s, edge_v, edge_index

        Returns:
            Drug representations of shape (batch_size, max_atoms, hidden_dim)
        """
        h_V = (xd.node_s, xd.node_v)
        h_E = (xd.edge_s, xd.edge_v)
        edge_index = xd.edge_index

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        out = self.W_out(h_V)
        out = self.pyg_split(xd, out)
        return out


class ConvEmbedding(nn.Module):
    """
    Convolutional embedding layer for sequence encoding.

    Applies multiple 1D convolutions with different kernel sizes to capture
    local patterns in sequences.

    Args:
        vocab_size: Size of vocabulary
        embedding_size: Embedding dimension
        conv_filters: List of [kernel_size, out_channels] pairs
        output_dim: Output dimension
        embed_type: 'seq' for sequence or 'poc' for pocket (with padding)
    """

    def __init__(self, vocab_size: int, embedding_size: int,
                 conv_filters: list, output_dim: int, embed_type: str):
        super().__init__()

        if embed_type == 'seq':
            self.embed = nn.Embedding(vocab_size, embedding_size)
        elif embed_type == 'poc':
            self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        else:
            self.embed = nn.Embedding(vocab_size, embedding_size)

        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in conv_filters:
            conv = nn.Conv1d(embedding_size, out_channels, kernel_size,
                           padding=(kernel_size - 1) // 2)
            self.convolutions.append(conv)

        self.num_filters = sum([f[1] for f in conv_filters])
        self.projection = nn.Linear(self.num_filters, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Input tensor of shape (batch_size, seq_len)

        Returns:
            Encoded tensor of shape (batch_size, seq_len, output_dim)
        """
        embeds = self.embed(inputs).transpose(-1, -2)  # (batch, embedding, seq_len)
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(embeds))
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim=1).transpose(-1, -2)
        embeds = self.projection(res_embed)
        return embeds


# Default convolution filter configuration
CONV_FILTERS = [[1, 32], [3, 32], [5, 64], [7, 128]]


class SeqEncoder(nn.Module):
    """
    Protein sequence encoder using Mamba2 State Space Model.

    Encodes both full protein sequence and pocket sequence, then combines
    global and local features.

    Args:
        embedding_dim: Dimension of sequence embeddings
        d_state: State dimension for Mamba2
        d_conv: Convolution dimension for Mamba2
        expand: Expansion factor for Mamba2
    """

    def __init__(self, embedding_dim: int = 256,
                 d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()

        self.seq_emb = ConvEmbedding(
            vocab_size=27, embedding_size=embedding_dim,
            conv_filters=CONV_FILTERS, output_dim=embedding_dim, embed_type='seq'
        )
        self.poc_emb = ConvEmbedding(
            vocab_size=27, embedding_size=embedding_dim,
            conv_filters=CONV_FILTERS, output_dim=embedding_dim, embed_type='poc'
        )

        if HAS_MAMBA:
            self.mamba2_glo = Mamba2(d_model=embedding_dim, d_state=d_state,
                                     d_conv=d_conv, expand=expand)
            self.mamba2_loc = Mamba2(d_model=embedding_dim, d_state=d_state,
                                     d_conv=d_conv, expand=expand)
        else:
            # Fallback to LSTM if Mamba is not available
            self.mamba2_glo = nn.LSTM(embedding_dim, embedding_dim // 2,
                                      batch_first=True, bidirectional=True)
            self.mamba2_loc = nn.LSTM(embedding_dim, embedding_dim // 2,
                                      batch_first=True, bidirectional=True)

        self.linear = nn.Linear(embedding_dim * 2, embedding_dim // 2)

    def forward(self, seq_input: torch.Tensor, poc_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_input: Full protein sequence tensor (batch, seq_len)
            poc_input: Pocket sequence tensor (batch, seq_len)

        Returns:
            Combined sequence features (batch, seq_len, embedding_dim // 2)
        """
        global_emb = self.seq_emb(seq_input)

        if HAS_MAMBA:
            global_feats = self.mamba2_glo(F.relu(global_emb))
        else:
            global_feats, _ = self.mamba2_glo(F.relu(global_emb))

        local_emb = self.poc_emb(poc_input)

        if HAS_MAMBA:
            local_feats = self.mamba2_loc(F.relu(local_emb))
        else:
            local_feats, _ = self.mamba2_loc(F.relu(local_emb))

        output = torch.cat((global_feats, local_feats), dim=-1)
        output = F.relu(self.linear(output))
        return output


class SmileEncoder(nn.Module):
    """
    SMILES sequence encoder using Mamba2 State Space Model.

    Encodes molecular SMILES strings into continuous representations.

    Args:
        embedding_dim: Dimension of SMILES embeddings
        d_state: State dimension for Mamba2
        d_conv: Convolution dimension for Mamba2
        expand: Expansion factor for Mamba2
    """

    def __init__(self, embedding_dim: int = 256,
                 d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()

        self.seq_emb = ConvEmbedding(
            vocab_size=86, embedding_size=embedding_dim,
            conv_filters=CONV_FILTERS, output_dim=embedding_dim, embed_type='seq'
        )

        if HAS_MAMBA:
            self.mamba2 = Mamba2(d_model=embedding_dim, d_state=d_state,
                                d_conv=d_conv, expand=expand)
        else:
            self.mamba2 = nn.LSTM(embedding_dim, embedding_dim // 2,
                                 batch_first=True, bidirectional=True)

        self.linear = nn.Linear(embedding_dim, embedding_dim // 2)

    def forward(self, smile_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            smile_input: SMILES sequence tensor (batch, seq_len)

        Returns:
            SMILES features (batch, seq_len, embedding_dim // 2)
        """
        emb = self.seq_emb(smile_input)

        if HAS_MAMBA:
            feats = self.mamba2(F.relu(emb))
        else:
            feats, _ = self.mamba2(F.relu(emb))

        output = F.relu(self.linear(feats))
        return output
