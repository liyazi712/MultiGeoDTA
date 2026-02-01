"""
Geometric Vector Perceptron (GVP) Module

Implementation of GVP layers for SE(3)-equivariant neural networks on molecular structures.
Adapted from: https://github.com/drorlab/gvp-pytorch

Reference:
    Jing, B., Eismann, S., Suriana, P., Townshend, R. J., & Dror, R. (2020).
    Learning from Protein Structure with Geometric Vector Perceptrons.
    arXiv preprint arXiv:2009.01411.
"""

import torch
import functools
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from typing import Tuple, Optional, List


def tuple_sum(*args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Element-wise sum of tuples (s, V).

    Args:
        *args: Variable number of tuples (scalar_tensor, vector_tensor)

    Returns:
        Tuple of summed (scalar, vector) tensors
    """
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Element-wise concatenation of tuples (s, V).

    Args:
        *args: Variable number of tuples (scalar_tensor, vector_tensor)
        dim: Dimension along which to concatenate

    Returns:
        Tuple of concatenated (scalar, vector) tensors
    """
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x: Tuple[torch.Tensor, torch.Tensor], idx) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Index into a tuple (s, V) along the first dimension.

    Args:
        x: Tuple of (scalar, vector) tensors
        idx: Index object for tensor indexing

    Returns:
        Indexed tuple (scalar, vector)
    """
    return x[0][idx], x[1][idx]


def randn(n: int, dims: Tuple[int, int], device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random tuples (s, V) from normal distribution.

    Args:
        n: Number of data points
        dims: Tuple of (n_scalar, n_vector) dimensions

    Returns:
        Tuple (s, V) with s.shape=(n, n_scalar) and V.shape=(n, n_vector, 3)
    """
    return (
        torch.randn(n, dims[0], device=device),
        torch.randn(n, dims[1], 3, device=device)
    )


def _norm_no_nan(x: torch.Tensor, axis: int = -1, keepdims: bool = False,
                  eps: float = 1e-8, sqrt: bool = True) -> torch.Tensor:
    """
    L2 norm of tensor clamped above minimum value.

    Args:
        x: Input tensor
        axis: Dimension to compute norm over
        keepdims: Whether to keep reduced dimensions
        eps: Minimum clamp value
        sqrt: If False, returns squared L2 norm

    Returns:
        L2 norm (or squared norm) tensor
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x: torch.Tensor, nv: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split merged representation back into tuple (s, V).

    Args:
        x: Merged tensor from _merge()
        nv: Number of vector channels

    Returns:
        Tuple (scalar, vector) tensors
    """
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v


def _merge(s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Merge tuple (s, V) into single tensor.

    Args:
        s: Scalar tensor
        v: Vector tensor of shape (..., n_vector, 3)

    Returns:
        Merged tensor with flattened vector channels
    """
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)


class GVP(nn.Module):
    """
    Geometric Vector Perceptron layer.

    Processes tuples of scalar and vector features in an SE(3)-equivariant manner.

    Args:
        in_dims: Tuple (n_scalar_in, n_vector_in)
        out_dims: Tuple (n_scalar_out, n_vector_out)
        h_dim: Intermediate vector channels (optional)
        activations: Tuple (scalar_activation, vector_activation)
        vector_gate: Whether to use vector gating mechanism
    """

    def __init__(self, in_dims: Tuple[int, int], out_dims: Tuple[int, int],
                 h_dim: Optional[int] = None,
                 activations: Tuple = (F.relu, torch.sigmoid),
                 vector_gate: bool = False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate

        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tuple (s, V) or single tensor if no vector input

        Returns:
            Tuple (s, V) or single tensor if no vector output
        """
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)

        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    """
    Vector channel dropout where elements of each vector channel are dropped together.
    """

    def __init__(self, drop_rate: float):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    """
    Combined dropout for tuples (s, V).
    """

    def __init__(self, drop_rate: float):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    """

    def __init__(self, dims: Tuple[int, int]):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class GVPConv(MessagePassing):
    """
    Graph convolution with Geometric Vector Perceptrons.

    Args:
        in_dims: Input node embedding dimensions (n_scalar, n_vector)
        out_dims: Output node embedding dimensions
        edge_dims: Edge embedding dimensions
        n_layers: Number of GVPs in message function
        aggr: Aggregation method ('add' or 'mean')
        activations: Activation functions tuple
        vector_gate: Whether to use vector gating
    """

    def __init__(self, in_dims: Tuple[int, int], out_dims: Tuple[int, int],
                 edge_dims: Tuple[int, int], n_layers: int = 1,
                 module_list: Optional[List] = None, aggr: str = "mean",
                 activations: Tuple = (F.relu, torch.sigmoid),
                 vector_gate: bool = False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve),
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims))
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        x_s, x_v = x
        message = self.propagate(edge_index,
                    s=x_s, v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                    edge_attr=edge_attr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    """
    Full GVP convolution layer with residual updates and feedforward network.

    Args:
        node_dims: Node embedding dimensions (n_scalar, n_vector)
        edge_dims: Edge embedding dimensions
        n_message: Number of GVPs in message function
        n_feedforward: Number of GVPs in feedforward function
        drop_rate: Dropout probability
        autoregressive: Whether to use autoregressive masking
        activations: Activation functions
        vector_gate: Whether to use vector gating
    """

    def __init__(self, node_dims: Tuple[int, int], edge_dims: Tuple[int, int],
                 n_message: int = 3, n_feedforward: int = 2, drop_rate: float = 0.1,
                 autoregressive: bool = False,
                 activations: Tuple = (F.relu, torch.sigmoid),
                 vector_gate: bool = False):
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )

            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)
        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x
