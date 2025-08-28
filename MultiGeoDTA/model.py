import torch
import torch.nn as nn
from MultiGeoDTA.gvp import GVP, GVPConvLayer, LayerNorm
import torch.nn.functional as F
from mamba_ssm import Mamba2


class ProtGVPModel(nn.Module):
    def __init__(self,
                 node_in_dim=None, node_h_dim=None,
                 edge_in_dim=None, edge_h_dim=None,
                 num_layers=None, drop_rate=0.1
                 ):
        """
        Parameters
        ----------
        node_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        node_h_dims : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        """
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
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def pyg_split(self, batched_data, feats):
        device = feats.device
        batch_size = batched_data.ptr.size(0) - 1
        node_to_graph_idx = batched_data.batch
        num_nodes_per_graph = torch.bincount(node_to_graph_idx)
        max_num_nodes = int(num_nodes_per_graph.max())
        batch = torch.cat(
            [torch.full((1, x.type(torch.int)), y) for x, y in zip(num_nodes_per_graph, range(batch_size))],
            dim=1).reshape(-1).type(torch.long).to(device)
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes_per_graph.cumsum(dim=0)])
        idx = torch.arange(len(node_to_graph_idx), dtype=torch.long, device=device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def forward(self, xp):
        # Unpack input data
        h_V = (xp.node_s, xp.node_v)
        h_E = (xp.edge_s, xp.edge_v)
        edge_index = xp.edge_index

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        out = self.pyg_split(xp, out)
        # out = torch_geometric.nn.global_add_pool(out, batch)

        return out


class DrugGVPModel(nn.Module):
    def __init__(self,
                 node_in_dim=None, node_h_dim=None,
                 edge_in_dim=None, edge_h_dim=None,
                 num_layers=None, drop_rate=0.1
                 ):
        """
        Parameters
        ----------
        node_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        node_h_dim : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        edge_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        edge_h_dim : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        """
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
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def pyg_split(self, batched_data, feats):
        # print(batched_data)
        # print(batched_data.batch)
        device = feats.device
        batch_size = batched_data.ptr.size(0) - 1
        node_to_graph_idx = batched_data.batch
        num_nodes_per_graph = torch.bincount(node_to_graph_idx)
        max_num_nodes = int(num_nodes_per_graph.max())

        batch = torch.cat(
            [torch.full((1, x.type(torch.int)), y) for x, y in zip(num_nodes_per_graph, range(batch_size))],
            dim=1).reshape(-1).type(torch.long).to(device)
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes_per_graph.cumsum(dim=0)])
        idx = torch.arange(len(node_to_graph_idx), dtype=torch.long, device=device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([batch_size, max_num_nodes] + list(feats.size())[1:])
        return out

    def forward(self, xd):
        # Unpack input data
        h_V = (xd.node_s, xd.node_v)
        # print(xp.node_s.shape, xp.node_v.shape, xp.edge_s.shape, xp.edge_v.shape)
        h_E = (xd.edge_s, xd.edge_v)
        edge_index = xd.edge_index
        batch = xd.batch

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)

        out = self.W_out(h_V)
        out = self.pyg_split(xd, out)
        return out


class ConvEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, conv_filters, output_dim, type):
        super().__init__()
        if type == 'seq':
            self.embed = nn.Embedding(vocab_size, embedding_size)

        elif type == 'poc':
            self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in conv_filters:
            conv = nn.Conv1d(embedding_size, out_channels, kernel_size, padding = (kernel_size - 1) // 2)
            self.convolutions.append(conv)
        # The dimension of concatenated vectors obtained from multiple one-dimensional convolutions
        self.num_filters = sum([f[1] for f in conv_filters])
        self.projection = nn.Linear(self.num_filters, output_dim)

    def forward(self, inputs):
        # embeds = self.embed(inputs)
        embeds = self.embed(inputs).transpose(-1,-2) # (batch_size, embedding_size, seq_len)
        conv_hidden = []
        for layer in self.convolutions:
            conv = F.relu(layer(embeds))
            conv_hidden.append(conv)
        res_embed = torch.cat(conv_hidden, dim = 1).transpose(-1,-2) # (batch_size, seq_len, num_filters)
        embeds = self.projection(res_embed)
        return embeds


conv_filters = [[1, 32], [3, 32], [5, 64], [7, 128]]
class Seq_Encoder(nn.Module):
    def __init__(self, embedding_dim=None):
        super().__init__()
        self.seq_emb = ConvEmbedding(vocab_size=27, embedding_size=embedding_dim,
                                     conv_filters=conv_filters, output_dim=embedding_dim, type='seq')
        self.poc_emb = ConvEmbedding(vocab_size=27, embedding_size=embedding_dim,
                                     conv_filters=conv_filters, output_dim=embedding_dim, type='poc')
        self.mamba2_glo = Mamba2(d_model=embedding_dim, d_state=64, d_conv=4, expand=2) # 256, 64 （128，64）, 4 (2～4), 2
        self.mamba2_loc = Mamba2(d_model=embedding_dim, d_state=64, d_conv=4, expand=2)
        self.linear = nn.Linear(embedding_dim * 2, embedding_dim // 2)

    def forward(self, seq_input, poc_input):
        global_emb = self.seq_emb(seq_input) # (128, 1024, 256)
        global_feats = self.mamba2_glo(F.relu(global_emb))  # (128, 1024, 256)

        local_emb = self.poc_emb(poc_input) # (128, 1024, 256)
        local_feats = self.mamba2_loc(F.relu(local_emb))  # (128, 1024, 256)

        output = torch.cat((global_feats, local_feats), dim=-1) # (128, 1024, 512)
        output = F.relu(self.linear(output)) # (128, 1024, 128)
        return output


class Smile_Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.seq_emb = ConvEmbedding(vocab_size=86, embedding_size=embedding_dim, conv_filters=conv_filters,
                                 output_dim=embedding_dim, type='seq')

        self.mamba2 = Mamba2(d_model=embedding_dim, d_state=64, d_conv=4, expand=2) # 256, 64 （128，64）, 4 (2～4), 2
        # d_model * expand / hidden_dim(default=64) == 8
        self.linear = nn.Linear(embedding_dim, embedding_dim // 2)

    def forward(self, smile_input):
        emb = self.seq_emb(smile_input) # (128, 1024, 256)
        feats = self.mamba2(F.relu(emb))  # (128, 1024, 256)
        output = F.relu(self.linear(feats))
        return output


class DTAModel(nn.Module):
    def __init__(self,
                 drug_node_in_dim=[86, 1], drug_node_h_dims=[128, 64],
                 drug_edge_in_dim=[24, 3], drug_edge_h_dims=[32, 1],
                 prot_node_in_dim=[6, 3], prot_node_h_dims=[128, 64],
                 prot_edge_in_dim=[32, 1], prot_edge_h_dims=[32, 1],
                 mlp_dims=[1024, 512], mlp_dropout=0.25):
        super(DTAModel, self).__init__()

        self.drug_GVP = DrugGVPModel(
            node_in_dim=drug_node_in_dim, node_h_dim=drug_node_h_dims,
            edge_in_dim=drug_edge_in_dim, edge_h_dim=drug_edge_h_dims,
            num_layers=1
        )

        self.prot_GVP = ProtGVPModel(
            node_in_dim=prot_node_in_dim, node_h_dim=prot_node_h_dims,
            edge_in_dim=prot_edge_in_dim, edge_h_dim=prot_edge_h_dims,
            num_layers=3
        )

        hidden_dim = prot_node_h_dims[0]
        self.seq_encoder = Seq_Encoder(embedding_dim=256)  # 256
        self.smile_encoder = Smile_Encoder(embedding_dim=256)

        # 128*3, 1024, 512, 1
        self.mlp = self.get_fc_layers(
            [hidden_dim * 4] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

    def get_fc_layers(self, hidden_sizes,
                      dropout=0, batchnorm=False,
                      no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.ReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        return nn.Sequential(*layers)


    def forward(self, xd, xp, protein_seq, pocket_seq, smile_seq):

        protein_feats = torch.mean(self.prot_GVP(xp), dim=1) # [batch_size, hidden_dim] (128, 128)
        seq_feats = torch.mean(self.seq_encoder(protein_seq, pocket_seq), dim=1) # [batch_size, hidden_dim] (128, 128)
        compound_feats = torch.mean(self.drug_GVP(xd), dim=1)  # [batch_size, hidden_dim] (128, 128)
        smile_feats = torch.mean(self.smile_encoder(smile_seq), dim=1) # [batch_size, hidden_dim] (128, 128)

        combined_feats = torch.cat([protein_feats, seq_feats, compound_feats, smile_feats], dim=-1)
        x = self.mlp(combined_feats)

        return x, protein_feats, seq_feats, compound_feats, smile_feats

