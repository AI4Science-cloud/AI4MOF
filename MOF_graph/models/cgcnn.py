import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    CGConv,
    #SAGPooling,
)
#from matdeeplearn.models.att_pool import GINATTPooling
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter


# CGCNN
class CGCNN(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        num_edge_features=3,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(CGCNN, self).__init__()
        in_features_edges = 3
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"        
        if pre_fc_count == 0:
            gc_dim = data.num_features #data.num_edge_features: 50 data.num_features: 114
        else:
            gc_dim = dim1#64
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1#64
        ##Determine output dimension length
        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
        #print(pre_fc_count)
        if pre_fc_count > 0:
            #print(pre_fc_count)
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):#1
                if i == 0:
                    lin = torch.nn.Linear(data.num_features, dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        ##Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        self.pool_list = torch.nn.ModuleList()
        for i in range(gc_count):#3
            conv = CGConv(
                gc_dim, num_edge_features, aggr="mean", batch_norm=False
            )
            self.conv_list.append(conv)
            
          
            if self.batch_norm == "True":
                bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)
       
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            #self.lin_out = torch.nn.Linear(dim2+2, output_dim)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim*2, output_dim)
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)   

        ##Set up set2set pooling (if used)
        ##Should processing_setps be a hypereparameter?
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)

    def forward(self, data):
        #print(data)
        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                #print(len(self.pre_lin_list))
                #print(len(data.x))
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        ##GNN layers
        #print(type(data.edge_index[0]))
        #print(isinstance(data.edge_index[0],torch.tensor))
        #print(hasattr(data,'edge_index_1'))
        
        if hasattr(data,'edge_index_1'):
            edge_index = torch.cat([data.edge_index_1, data.edge_index_2, data.edge_index_3], 1)
            edge_attr = torch.cat([data.edge_attr_1, data.edge_attr_2, data.edge_attr_3], 0)
            
        else:
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            
        #edge_attr = data.edge_attr_1
        #print(data.edge_attr.shape)
        #print(edge_index.shape, edge_attr.shape)
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    out = self.conv_list[i](data.x, edge_index, edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](data.x, edge_index, edge_attr)
            else:
                if self.batch_norm == "True":
                    #print(out.shape,edge_index.shape,edge_attr.shape)
                    out = self.conv_list[i](out, edge_index, edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](out, edge_index, edge_attr)            
            #out = getattr(F, self.act)(out)
            #out, edge_index, edge_attr, batch, _, _ = self.pool_list[i](out, edge_index, edge_attr, data.batch)
            
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        #out, edge_index, batch, _ = self.pool_edge(out, edge_index, data.batch)
	
        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out,data.batch)
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            #parameters = torch.reshape(data.parameters,(-1,2))
            #out = torch.cat([out,parameters],1)
            out = self.lin_out(out)

        elif self.pool_order == "late":
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out
