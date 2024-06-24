"""Implementation based on the template of ALIGNN."""

from typing import Tuple
from typing import Optional
import math
import numpy as np
import torch
import torch.nn.functional as F
from pydantic.typing import Literal
from torch import nn
#from matformer.models.utils import RBFExpansion

# from matformer.utils import BaseSettings
# from matformer.features import angle_emb_mp
from torch_scatter import scatter
#from matformer.models.transformer import MatformerConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor

#import torch
#import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn as nn
from typing import Optional, Tuple, Union

class MultiFormer(torch.nn.Module):
    """att pyg implementation."""

    def __init__(
        self, 
        data,
        dim1=100,
        dim2=150,
        dim3=128,
        output_features=1,
        pre_fc_count=1,
        gc_count=3,
        node_layer_head=4,
        edge_layer_head=4,
        conv_layers=3,
        edge_layers=0,
        #bias=True,
        # assign_count=1,
        # embedd_count=1,
        diff_count=1,
        post_fc_count=1,
        pool_count = 3,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        """Set up att modules."""
        super().__init__()
        # self.classification = config.classification
        # self.use_angle = config.use_angle
        
        self.atom_embedding = nn.Linear(
            data.num_features, dim1
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=data.num_edge_features,
            ),
            nn.Linear(data.num_edge_features, data.num_features),
            nn.Softplus(),
            nn.Linear(data.num_features, data.num_features),
        )

        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=int(dim1), out_channels=int(dim1), heads=int(node_layer_head), edge_dim=93)#node_features)
                for _ in range(conv_layers)
            ]
        )
        
        self.edge_update_layers = nn.ModuleList( ## module not used
            [
                MatformerConv(in_channels=int(dim1), out_channels=int(dim1), heads=int(edge_layer_head), edge_dim=93)
                for _ in range(edge_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(dim1, dim2), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

        # if self.classification:
        #     self.fc_out = nn.Linear(config.fc_features, 2)
        #     self.softmax = nn.LogSoftmax(dim=1)
        # else:
        self.fc_out = nn.Linear(dim2, output_features)

        # self.link = None
        # self.link_name = config.link
        # if config.link == "identity":
        self.link = lambda x: x
        # elif config.link == "log":
        #     self.link = torch.exp
        #     avg_gap = 0.7  # magic number -- average bandgap in dft_3d
        #     if not self.zero_inflated:
        #         self.fc_out.bias.data = torch.tensor(
        #             np.log(avg_gap), dtype=torch.float
        #         )
        # elif config.link == "logit":
        #     self.link = torch.sigmoid

    def forward(self, data) -> torch.Tensor:
        # data, ldata, lattice = data
        # initial node features: atom feature network...
            
        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)
        
        edge_features = self.rbf(edge_feat)
      
        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        # node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        # node_features = self.att_layers[4](node_features, data.edge_index, edge_features)


        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

       
        features = self.fc(features)

        out = self.fc_out(features)
        #if self.link:
        out = self.link(out)
        # print(out)
        # print(torch.squeeze(out))
        return torch.squeeze(out)




class MatformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(MatformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)
        
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], out_channels,
                                   bias=bias)
            self.lin_concate = nn.Linear(heads * out_channels, out_channels)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.msg_layer = nn.Sequential(nn.Linear(out_channels * 3, out_channels), nn.LayerNorm(out_channels))
        # self.msg_layer = nn.Linear(out_channels * 3, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        # self.bn = nn.BatchNorm1d(out_channels * heads)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.concat:
            self.lin_concate.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.concat:
            out = self.lin_concate(out)

        out = F.silu(self.bn(out)) # after norm and silu

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
        query_i = torch.cat((query_i, query_i, query_i), dim=-1)
        key_j = torch.cat((key_i, key_j, edge_attr), dim=-1)
        alpha = (query_i * key_j) / math.sqrt(self.out_channels * 3) 
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.cat((value_i, value_j, edge_attr), dim=-1)
        out = self.lin_msg_update(out) * self.sigmoid(self.layer_norm(alpha.view(-1, self.heads, 3 * self.out_channels))) 
        out = self.msg_layer(out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )

