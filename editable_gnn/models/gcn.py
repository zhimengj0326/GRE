from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from .base import BaseModel


class GCN(BaseModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False):
        super(GCN, self).__init__(in_channels, hidden_channels, out_channels, 
                                  num_layers, dropout, batch_norm, residual)
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
            conv = GCNConv(in_dim, out_dim, normalize=False)
            self.convs.append(conv)