from typing import Callable, Optional, Union

import torch

from torch_geometric.nn import GINEConv,CGConv#, Sequential
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax


class GINATTPooling(torch.nn.Module):
  
    #def __init__(self, in_channels: int, hidden_channels: int, edge_dim: int, ratio: Union[float, int] = 0.5,
                 #GINE: Callable = GINEConv, min_score: Optional[float] = None,
                 #multiplier: float = 1.0, nonlinearity: Callable = torch.tanh,
                 #**kwargs):
    def __init__(self, in_channels: int, hidden_channels: int, edge_dim: int, ratio: Union[float, int] = 0.5,
                 CGCNN: Callable = CGConv, min_score: Optional[float] = None,
                 multiplier: float = 1.0, nonlinearity: Callable = torch.tanh,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.ratio = ratio
        self.edge_dim = edge_dim
        #nnl =  Sequential(Linear(self.in_channels, self.hidden_channels), ReLU(), Linear(self.hidden_channels,1))
        #self.gin = GINE(nn=nnl, edge_dim = self.edge_dim, **kwargs)
        self.cgcnn = CGCNN(self.in_channels, self.edge_dim,aggr="mean",batch_norm=False)
        self.lin_score = Linear(self.in_channels, 1, bias=True)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity
        self.select = SelectTopK(1, ratio, min_score, nonlinearity)
        self.connect = FilterEdges()

        self.reset_parameters()

    def reset_parameters(self):
        #self.gin.reset_parameters()
        self.cgcnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch=None, attn=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.view(-1, 1) if attn.dim() == 1 else attn
        #attn = self.gin(attn, edge_index, edge_attr).view(-1)
        attn = self.lin_score(self.cgcnn(attn, edge_index, edge_attr)).view(-1)
       
        select_out = self.select(attn, batch)

        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        x = x[perm] * score.view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        connect_out = self.connect(select_out, edge_index, edge_attr, batch)

        return (x, connect_out.edge_index, connect_out.edge_attr,
                connect_out.batch, perm, score)
    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'
        return (f'{self.__class__.__name__}({self.cgcnn.__class__.__name__}, '
                f'{self.in_channels}, {ratio}, multiplier={self.multiplier})')
        #return (f'{self.__class__.__name__}({self.gin.__class__.__name__}, '
        #        f'{self.in_channels}, {ratio}, multiplier={self.multiplier})')
