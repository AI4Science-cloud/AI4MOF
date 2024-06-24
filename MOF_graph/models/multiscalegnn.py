import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d
import torch_geometric
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    CGConv,
)
from torch_geometric.nn import SAGPooling
import numpy as np
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
#from collections import Counter
from torch_geometric.nn import GINEConv, global_add_pool    
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from MOF_graph.models.att_pool import GINATTPooling
from MOF_graph.models.edge_pool import EdgePooling
# CGCNN
class MultiScaleConvNet(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=100,
        dim2=128,
        pre_fc_count=4,#1,
        # gc_count=3,
        n_output=1,
        #bias=True,
        # assign_count=1,
        # embedd_count=1,
        # diff_count=1,
        # post_fc_count=1,
        # pool_count = 2,
        edge_attr_size = 3,
        pool="global_mean_pool",
        pool_mean="global_mean_pool",
        pool1="global_add_pool",
        #pool_order="early",
        # batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout=0.2,
        **kwargs
    ):
        super(MultiScaleConvNet, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.n_output = n_output#1#len(data.y)
        self.batch_track_stats = batch_track_stats
        self.pool  = pool 
        self.pool_mean = pool_mean
        self.pool1 = pool1
        in_features_xd = data.num_features
        in_features_edges = edge_attr_size
        # self.mlp = nn.Linear(2 * output_dim, output_dim)
        # self.leakyrelu = nn.LeakyReLU()
        # self.softmax = nn.Softmax(dim=1)
        
        if pre_fc_count > 0:
            #print(pre_fc_count)
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):#1
                if i == 0:
                    lin = torch.nn.Linear(in_features_xd, dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()
        
    
        # scale_1 convolution layers
        
        self.conv1 = CGConv(dim1, in_features_edges, aggr="mean", batch_norm=False)
        self.bn1 = torch.nn.BatchNorm1d(dim1, track_running_stats=self.batch_track_stats)

        
        # scale_2 convolution layers
        
        self.conv1_2 = CGConv(dim1, in_features_edges, aggr="mean", batch_norm=False)
        self.bn1_2 = torch.nn.BatchNorm1d(dim1, track_running_stats=self.batch_track_stats)

        
        self.conv1_3 = CGConv(dim1, in_features_edges, aggr="mean", batch_norm=False)
        self.bn1_3 = torch.nn.BatchNorm1d(dim1, track_running_stats=self.batch_track_stats)

       
        self.conv1_4 = CGConv(dim1, in_features_edges+1, aggr="mean", batch_norm=False)
        self.bn1_4 = torch.nn.BatchNorm1d(dim1, track_running_stats=self.batch_track_stats)
        self.conv2_4 = CGConv(dim1, in_features_edges+1, aggr="mean", batch_norm=False)
        self.bn2_4 = torch.nn.BatchNorm1d(dim1, track_running_stats=self.batch_track_stats)
        self.conv3_4 = CGConv(dim1, in_features_edges+1, aggr="mean", batch_norm=False)
        self.bn3_4 = torch.nn.BatchNorm1d(dim1, track_running_stats=self.batch_track_stats)
        
        self.att_pool_1 = EdgePooling(dim1, in_features_edges, dropout=0.2, ratio =1.0)
        self.att_pool_2 = EdgePooling(dim1, in_features_edges, dropout=0.2, ratio =0.9)
        self.att_pool_3 = EdgePooling(dim1, in_features_edges, dropout=0.2, ratio =0.8)
  
        #self.graphout = nn.Linear(dim1+2, n_output)
        self.graphout1 = nn.Linear(dim1+2, dim2)
        #self.graphout1 = nn.Linear(dim1, dim2)
        self.graphout2 = nn.Linear(dim2, n_output)
    

    def forward(self, data, batch_num_nodes=None):
        
        
       x, edge_index_1, edge_index_2, edge_index_3, edge_attr_1, edge_attr_2, edge_attr_3, batch = data.x, data.edge_index_1, data.edge_index_2, data.edge_index_3, data.edge_attr_1, data.edge_attr_2, data.edge_attr_3, data.batch
       
       ##Pre-GNN dense layers
       for i in range(0, len(self.pre_lin_list)):
           x = self.pre_lin_list[i](x)
           x = getattr(F, self.act)(x)
          
       x_1 = self.conv1(x, edge_index_1, edge_attr_1)
       x_1 = self.bn1(x_1)

       edge_index_1_, edge_attr_1_, score1 = self.att_pool_1(x_1, edge_index_1, edge_attr_1, batch)
       #print(edge_attr_1_.shape)
       edge_attr_1_ = torch.cat([edge_attr_1_, score1],dim=1)
       
       x_2 = self.conv1_2(x, edge_index_2, edge_attr_2)
       
       x_2 = self.bn1_2(x_2)
     
       edge_index_2_, edge_attr_2_, score2 = self.att_pool_1(x_2, edge_index_2, edge_attr_2, batch)
       edge_attr_2_ = torch.cat([edge_attr_2_, score2],dim=1)
     
       
       x_3 = self.conv1_3(x, edge_index_3, edge_attr_3)
       x_3 = self.bn1_3(x_3)
   
       edge_index_3_, edge_attr_3_, score3 = self.att_pool_1(x_3, edge_index_3, edge_attr_3, batch)
       edge_attr_3_ = torch.cat([edge_attr_3_, score3],dim=1)
       edge_index = torch.cat([edge_index_1_, edge_index_2_, edge_index_3_], 1) 
       edge_attr = torch.cat([edge_attr_1_, edge_attr_2_, edge_attr_3_], 0) 
       
       out = self.conv1_4(x, edge_index, edge_attr)
       out = self.bn1_4(out)
       out = self.conv2_4(out, edge_index, edge_attr)
       out = self.bn2_4(out)
       out = self.conv3_4(out, edge_index, edge_attr)
       out = self.bn3_4(out)
       parameters = torch.reshape(data.parameters,(-1,2))
       
       
    
       out = getattr(torch_geometric.nn, self.pool)(out, batch)
   
       out = torch.cat([out,parameters],1)
       
       
       
       out = self.graphout1(out)
       out = self.graphout2(out)
       
       draw = False
       
       if draw == True:
           edge_index_sum = torch.cat([edge_index_1, edge_index_2, edge_index_3], 1) 
           positions = data.position
           return edge_index_1_, edge_index_2_, edge_index_3_, score1, score2, score3, edge_index_sum, positions, batch, parameters, data.structure_id, data.x
       
       else:
           if out.shape[1] == 1:
               # print(out.view(-1))
               return out.view(-1)#, edge_attr, batch, edge_index
           else:
               # print(out)
               return out
