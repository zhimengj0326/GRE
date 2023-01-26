from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d, Linear
from torch_sparse import SparseTensor
from torch_geometric.nn import GCN2Conv



class GCN2(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float, theta: float = None, 
                 shared_weights: bool = True, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False):
        super(GCN2, self).__init__(in_channels, hidden_channels, out_channels, 
                                  num_layers, dropout, batch_norm, residual)
        self.alpha, self.theta = alpha, theta

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            if theta is None:
                conv = GCN2Conv(hidden_channels, alpha=alpha, theta=None,
                                layer=None, shared_weights=shared_weights,
                                normalize=False, add_self_loops=False)
            else:
                conv = GCN2Conv(hidden_channels, alpha=alpha, theta=theta,
                                layer=i+1, shared_weights=shared_weights,
                                normalize=False, add_self_loops=False)
            self.convs.append(conv)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()


    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        x = x0 = self.activation(self.lins[0](x))
        x = self.dropout(x)
        for idx, conv in enumerate(self.convs[:-1]):
            h = conv(x, x0, adj_t)
            if self.batch_norm:
                h = self.bns[idx](h)
            if self.residual:
                h += x[:h.size(0)]
            x = self.activation(h)
            x = self.dropout(x)

        h = self.convs[-1](x, x0, adj_t)
        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual:
            h += x[:h.size(0)]
        x = self.activation(h)
        x = self.dropout(x)
        x = self.lins[1](x)
        return x


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0:
            x = x_0 = self.activation(self.lins[0](x))
            state['x_0'] = x_0[:adj_t.size(0)]
        x = self.dropout(x)
        h = self.convs[layer](x, state['x_0'], adj_t)
        if self.batch_norm:
            h = self.bns[layer](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        x = self.activation(h)
        if layer == self.num_layers - 1:
            x = self.dropout(x)
            x = self.lins[1](x)  
        return h