import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import InstanceNorm, LayerNorm, GraphNorm, global_mean_pool, global_add_pool, global_max_pool

from .gen_conv import GENConv
from torch_geometric.nn.conv import GATConv, MFConv, LEConv, PANConv, ChebConv
from torch_geometric.nn.models import GAT
from torch_geometric.nn.pool import TopKPooling

import numpy as np

class GNNModel(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, virtual_node, model_config):
        super(GNNModel, self).__init__()
        self.out_channels = model_config['out_channels']
        self.n_layers = model_config['n_layers']
        self.dropout_p = model_config['dropout_p']
        self.readout = model_config['readout']
        self.norm_type = model_config['norm_type']
        self.deepgcn_aggr = model_config['deepgcn_aggr']
        self.bn_input = model_config['bn_input']
        self.virtual_node = virtual_node
        self.conv_type = model_config['conv_type']
        self.skip_num = model_config['skip_num']
        self.topk = model_config.get('topk', False)
        self.edge_atten = model_config.get('edge_atten', False) # Additional learned parameters for weighting edge features during message passing
        self.out_dim = model_config.get('out_dim', 1)
        
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.edge_updaters = nn.ModuleList()
        
        if self.topk:
            assert self.skip_num == 1
            self.topk_layers = nn.ModuleList()
        
        channels = [self.out_channels, self.out_channels*2, self.out_channels] # Dimensions of the MLPs in the hidden layers. Number of layers in MLP = len(channels)
        
        #self.input_drop = nn.Dropout(.1)
        
        self.node_encoder = Encoder(x_dim, self.out_channels) 
        
        self.edge_encoder = Encoder(edge_attr_dim, self.out_channels)

        if self.bn_input:
            self.bn_node_feature = nn.BatchNorm1d(self.out_channels)
            self.bn_edge_feature = nn.BatchNorm1d(self.out_channels)
        
        #print('Convolution Type: ', self.conv_type)
        if self.conv_type == 'GATConv':
            for _ in range(self.n_layers):
                self.convs.append(GATConv(self.out_channels, self.out_channels, edge_dim=self.out_channels, add_self_loops=False))
                self.mlps.append(MLP(channels, norm_type=self.norm_type, dropout=self.dropout_p))
                self.edge_updaters.append(MLP(channels, norm_type=self.norm_type, dropout=self.dropout_p))
        
        elif self.conv_type == 'MFConv':
            for _ in range(self.n_layers):
                self.convs.append(MFConv(self.out_channels, self.out_channels))
                self.mlps.append(MLP(channels, norm_type=self.norm_type, dropout=self.dropout_p))
        
        elif self.conv_type == 'LEConv':
            for _ in range(self.n_layers):
                self.convs.append(LEConv(self.out_channels, self.out_channels))
                self.mlps.append(MLP(channels, norm_type=self.norm_type, dropout=self.dropout_p))
        
        elif self.conv_type == 'PANConv':
            for _ in range(self.n_layers):
                self.convs.append(PANConv(self.out_channels, self.out_channels, filter_size=2))
                self.mlps.append(MLP(channels, norm_type=self.norm_type, dropout=self.dropout_p))
                
        elif self.conv_type == 'ChebConv':
            for _ in range(self.n_layers):
                self.convs.append(ChebConv(self.out_channels, self.out_channels, K=1))
                self.mlps.append(MLP(channels, norm_type=self.norm_type, dropout=self.dropout_p))
                
        else: # GENConv
            for _ in range(self.n_layers):
                self.convs.append(GENConv(self.out_channels, self.out_channels, aggr=self.deepgcn_aggr, learn_t=True, learn_p=True, edge_atten=self.edge_atten))
                self.mlps.append(MLP(channels, norm_type=self.norm_type, dropout=self.dropout_p))
                self.edge_updaters.append(MLP(channels, norm_type=self.norm_type, dropout=self.dropout_p))
                
                if self.topk:
                    self.topk_layers.append(TopKPooling(self.out_channels, ratio=self.topk))
        
        if self.virtual_node:
            if self.readout == 'lstm':
                self.lstm = nn.LSTMCell(self.out_channels, self.out_channels)
            elif self.readout == 'jknet':
                self.downsample = nn.Linear(self.out_channels * self.n_layers, self.out_channels)
            elif self.readout == 'vn':
                pass
            elif self.readout == 'pool':
                self.pool = global_add_pool
            else:
                raise NotImplementedError
        else:
            assert self.readout == 'pool'
            self.pool = global_mean_pool
            
        self.out = Out(self.out_channels, self.out_dim)
        #self.out = nn.Linear(self.out_channels, 1)
        
        #self.node_clf_out = nn.Linear(self.out_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch, ptr, edge_atten=None, node_atten=None, node_clf=False):
        
        v_idx, v_emb = (ptr[1:] - 1, []) if self.virtual_node else (None, None)
        #x = self.input_drop(x)
        #edge_attr = self.input_drop(edge_attr)
        
        x = self.node_encoder(x, batch) # Encode node features
        edge_attr = self.edge_encoder(edge_attr, batch) # Encode edge features
    
        if node_atten is not None:
            assert edge_atten is not None
            x = node_atten * x
            edge_attr = edge_atten * edge_attr

        if self.bn_input:
            x = self.bn_node_feature(x)
            edge_attr = self.bn_edge_feature(edge_attr)
        
        stored_embs = [x] # Store embeddings for residual/skip connections
        for i in range(self.n_layers):
            j = i+1 # for indexing stored_embs
            if i < self.skip_num:
                skip = sum(stored_embs[:j])
            else:
                skip = sum(stored_embs[j-self.skip_num:j])
            
            if self.conv_type in ['MFConv', 'LEConv', 'ChebConv']: # No edge features
                x = self.convs[i](x, edge_index)
                x = self.mlps[i](x, batch)
            elif self.conv_type == 'PANConv': # No edge features
                x = self.convs[i](x, edge_index)[0]
                x = self.mlps[i](x, batch)
            else: # Use edge features
                x = self.convs[i](x, edge_index, edge_attr=edge_attr) # Update node embeddings with message passing
                x = self.mlps[i](x, batch) # Update node embeddings with MLP
                edge_attr = self.edge_updaters[i](edge_attr, batch) # Update edge embeddings with MLP
                
            if self.virtual_node:
                if i == 0 and self.readout != 'pool':
                    hx, cx = stored_embs[0][v_idx], torch.zeros_like(stored_embs[0][v_idx])
                if self.readout == 'lstm':
                    hx, cx = self.lstm(x[v_idx], (hx, cx))
                elif self.readout == 'jknet':
                    v_emb.append(x[v_idx])
                    
            x += skip # Residual/skip connection
            if self.topk:
                x, edge_index, edge_attr, batch, _, _ = self.topk_layers[i](x, edge_index, edge_attr=edge_attr, batch=batch)
            
            stored_embs.append(x)
        
        if not(node_clf):
            if self.virtual_node:
                if self.readout == 'lstm':
                    pool_out = hx
                elif self.readout == 'jknet':
                    pool_out = self.downsample(torch.cat(v_emb, dim=1))
                elif self.readout == 'vn':
                    pool_out = x[v_idx]
                elif self.readout == 'pool':

                    pool_out = self.pool(x, batch)
  
            else:
                pool_out = self.pool(x, batch)
            
            out = self.out(pool_out, batch)
        
        else:
            out = self.out(x)
        
        return out

    def get_emb(self, x, edge_index, edge_attr, batch, ptr):
        v_idx, v_emb = (ptr[1:] - 1, []) if self.virtual_node else (None, None)
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        if self.bn_input:
            x = self.bn_node_feature(x)
            edge_attr = self.bn_edge_feature(edge_attr)

        for i in range(self.n_layers):
            identity = x

            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.mlps[i](x, batch)

            if self.virtual_node:
                if i == 0:
                    hx, cx = identity[v_idx], torch.zeros_like(identity[v_idx])
                if self.readout == 'lstm':
                    hx, cx = self.lstm(x[v_idx], (hx, cx))
                elif self.readout == 'jknet':
                    v_emb.append(x[v_idx])
            x += identity
        return x


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (GraphNorm, InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, norm_type, dropout, bias=True):

        if norm_type == 'batch':
            norm = nn.BatchNorm1d
        elif norm_type == 'layer':
            norm = LayerNorm
        elif norm_type == 'instance':
            norm = InstanceNorm
        elif norm_type == 'graph':
            norm = GraphNorm
        else:
            raise NotImplementedError

        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(norm(channels[i]))
                m.append(nn.LeakyReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


class Encoder(BatchSequential): # MLP for node/edge encoding
    def __init__(self, in_channels, out_channels, n_layers=3, dropout=0, bias=True):

        m = []
        
        m.append(nn.Linear(in_channels, out_channels, bias))
        m.append(nn.LeakyReLU())
        m.append(nn.Dropout(dropout))
        
        for i in range(n_layers-1):
            m.append(nn.Linear(out_channels, out_channels, bias))
            m.append(nn.LeakyReLU())
            m.append(nn.Dropout(dropout))
        

        super(Encoder, self).__init__(*m)


class Out(BatchSequential): # MLP for output
    def __init__(self, in_channels, out_channels, n_layers=3, dropout=0, bias=True):
        
        m = []
        
        for i in range(n_layers-1):
            m.append(nn.Linear(in_channels, in_channels, bias))
            m.append(nn.LeakyReLU())
            m.append(nn.Dropout(dropout))
        
        m.append(nn.Linear(in_channels, out_channels, bias))
        
        
        super(Out, self).__init__(*m)
        
        
        
        
        
        
        
        
        
        
        
        
        