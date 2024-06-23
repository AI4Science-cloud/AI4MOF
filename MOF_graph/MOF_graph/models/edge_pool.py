from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional, Union
from torch_geometric.utils import coalesce, scatter, softmax
from torch_scatter import scatter_add, scatter_max

# class UnpoolInfo(NamedTuple):
#     edge_index: Tensor
#     cluster: Tensor
#     batch: Tensor
#     new_edge_score: Tensor


class EdgePooling(torch.nn.Module):
   
    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        edge_score_method: Optional[Callable] = None,
        dropout: Optional[float] = 0.0,
        add_to_edge_score: float = 0.5,
        ratio: Union[float, int] = 0.8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.edge_channls = edge_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout
        self.ratio = ratio
        self.lin_f = torch.nn.Linear(2 * in_channels + edge_channels, 1, bias=True)
        self.lin_s = torch.nn.Linear(2 * in_channels + edge_channels, 1, bias=True)
        self.lin = torch.nn.Linear(2 * in_channels + edge_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(
        raw_edge_score: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        #e, edge_index, x.size(0)
        r"""Normalizes edge scores via softmax application."""
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via hyperbolic tangent application."""
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via sigmoid application."""
        return torch.sigmoid(raw_edge_score)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ):
        r"""
        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(torch.Tensor)* - The pooled node features.
            * **edge_index** *(torch.Tensor)* - The coarsened edge indices.
            * **batch** *(torch.Tensor)* - The coarsened batch vector.
            * **unpool_info** *(UnpoolInfo)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        ratio = self.ratio 
        e = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=-1)
        #e = self.lin(e).view(-1)
        e = torch.mul(self.lin_f(e).sigmoid(), F.softplus(self.lin_s(e)))
        e = F.dropout(e, p=self.dropout, training=self.training)
        #e = self.compute_edge_score(e, edge_index, x.size(0))
        #e = e + self.add_to_edge_score

        # x, edge_index, batch, unpool_info = self._merge_edges(
        #     x, edge_index, batch, e)

        #return x, edge_index, batch, unpool_info

        edge_index, edge_attr, edge_batch, edge_score = self.choose_edges(edge_index, edge_attr, batch, e, ratio)
        
        return edge_index, edge_attr, edge_score
    
    def choose_edges(
        self,
        #x: Tensor,
        edge_index: Tensor,
        edge_attr:Tensor,
        batch: Tensor,
        edge_score: Tensor,
        ratio: Union[float, int] = 0.8,
            
    ):
        # num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        # batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
        # cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
        edge_batch = batch[edge_index[1]]
        num_edges = scatter_add(edge_batch.new_ones(edge_index[1].size(0)), edge_batch, dim=0)
        edge_batch_size, max_num_edges = num_edges.size(0), num_edges.max().item()
        cum_num_edges = torch.cat([num_edges.new_zeros(1), num_edges.cumsum(dim=0)[:-1]], dim=0)
        index = torch.arange(edge_batch.size(0), dtype=torch.long, device=edge_index.device)
        index = (index - cum_num_edges[edge_batch]) + (edge_batch * max_num_edges)
        dense_edge = edge_score.new_full((edge_batch_size * max_num_edges, 1), torch.finfo(edge_score.dtype).min)
        dense_edge[index] = edge_score
        dense_edge = dense_edge.view(edge_batch_size, max_num_edges)
        _, perm = dense_edge.sort(dim=-1, descending=True)
        perm = perm + cum_num_edges.view(-1, 1)
        perm = perm.view(-1)
        if isinstance(ratio, int):
            k = num_edges.new_full((num_edges.size(0), ), ratio)
            k = torch.min(k, num_edges)
        else:
            k = (ratio * num_edges.to(torch.float)).ceil().to(torch.long)
            
        mask =[
            torch.arange(k[i], dtype=torch.long, device=edge_index.device) +
            i * max_num_edges for i in range(edge_batch_size)#batch_size = 100
        ]
        mask = torch.cat(mask, dim=0)
        perm = perm[mask]
        
        edge_score = edge_score[perm]
        edge_attr = edge_attr[perm]
        edge_index = torch.stack([edge_index[0][perm], edge_index[1][perm]])
        edge_batch = edge_batch[perm]
        
        return edge_index, edge_attr, edge_batch, edge_score
    
    # def _merge_edges(
    #     self,
    #     x: Tensor,
    #     edge_index: Tensor,
    #     batch: Tensor,
    #     edge_score: Tensor,
    # ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:

    #     cluster = torch.empty_like(batch)
    #     perm: List[int] = torch.argsort(edge_score, descending=True).tolist()

    #     # Iterate through all edges, selecting it if it is not incident to
    #     # another already chosen edge.
    #     mask = torch.ones(x.size(0), dtype=torch.bool)

    #     i = 0
    #     new_edge_indices: List[int] = []
    #     edge_index_cpu = edge_index.cpu()
    #     for edge_idx in perm:
    #         source = int(edge_index_cpu[0, edge_idx])
    #         if not bool(mask[source]):
    #             continue

    #         target = int(edge_index_cpu[1, edge_idx])
    #         if not bool(mask[target]):
    #             continue

    #         new_edge_indices.append(edge_idx)

    #         cluster[source] = i
    #         mask[source] = False

    #         if source != target:
    #             cluster[target] = i
    #             mask[target] = False

    #         i += 1

    #     # The remaining nodes are simply kept:
    #     j = int(mask.sum())
    #     cluster[mask] = torch.arange(i, i + j, device=x.device)
    #     i += j

    #     # We compute the new features as an addition of the old ones.
    #     new_x = scatter(x, cluster, dim=0, dim_size=i, reduce='sum')
    #     new_edge_score = edge_score[new_edge_indices]
    #     if int(mask.sum()) > 0:
    #         remaining_score = x.new_ones(
    #             (new_x.size(0) - len(new_edge_indices), ))
    #         new_edge_score = torch.cat([new_edge_score, remaining_score])
    #     new_x = new_x * new_edge_score.view(-1, 1)

    #     new_edge_index = coalesce(cluster[edge_index], num_nodes=new_x.size(0))
    #     new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
    #     new_batch = new_batch.scatter_(0, cluster, batch)

    #     unpool_info = UnpoolInfo(edge_index=edge_index, cluster=cluster,
    #                              batch=batch, new_edge_score=new_edge_score)

    #     return new_x, new_edge_index, new_batch, unpool_info

    # def unpool(
    #     self,
    #     x: Tensor,
    #     unpool_info: UnpoolInfo,
    # ) -> Tuple[Tensor, Tensor, Tensor]:
    #     r"""Unpools a previous edge pooling step.

    #     For unpooling, :obj:`x` should be of same shape as those produced by
    #     this layer's :func:`forward` function. Then, it will produce an
    #     unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

    #     Args:
    #         x (torch.Tensor): The node features.
    #         unpool_info (UnpoolInfo): Information that has been produced by
    #             :func:`EdgePooling.forward`.

    #     Return types:
    #         * **x** *(torch.Tensor)* - The unpooled node features.
    #         * **edge_index** *(torch.Tensor)* - The new edge indices.
    #         * **batch** *(torch.Tensor)* - The new batch vector.
    #     """
    #     new_x = x / unpool_info.new_edge_score.view(-1, 1)
    #     new_x = new_x[unpool_info.cluster]
    #     return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'
