#!/usr/bin/env python
# encoding: utf-8
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Callable
from functools import reduce
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_add_pool as gsp

class FeatureEncoder(nn.Module):
    def __init__(self, args):
        super(FeatureEncoder, self).__init__()
        self.args = args
        self.feature_num = args.feature_num
        self.dropout = args.dropout
        input_dimension = args.input_dim
        embedding_dimension = args.hidden_dim

        self.linear_relu = nn.Sequential(
            nn.Linear(self.feature_num, input_dimension), 
            nn.LeakyReLU(), 
            nn.Linear(input_dimension, embedding_dimension),
            nn.PReLU(embedding_dimension)
        )

    def forward(self, x):
        x = self.linear_relu(x)
        return x


class SEPooling(MessagePassing):
    def __init__(self, nn: Callable, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, size=size)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class SEP_G(torch.nn.Module):
    def __init__(self, args):
        super(SEP_G, self).__init__()
        self.args = args
        self.num_features = args.input_dim  
        self.nhid = args.hidden_dim  
        self.num_classes = args.num_classes  
        self.num_features = args.num_features
        self.dropout_ratio = args.pooling_dropout
        self.feature_encoder = FeatureEncoder(args)
        self.convs = self.get_convs()
        self.sepools = self.get_sepool()
        self.global_pool = gsp if args.global_pooling == 'sum' else gap
        self.classifier = self.get_classifier()

    def __process_layer_edgeIndex(self, batch_data, layer = 0):
        edge_mat_list = []
        start_idx = [0]
        if layer == 0:
            for i, graph in enumerate(batch_data['knn_edge_index_list']):
                edge_mat_list.append(graph + self.args.attribute_num * i)
        else:
            for i, graph in enumerate(batch_data['layer_data']):
                start_idx.append(start_idx[i] + graph['node_size'][layer])
                edge_mat_list.append(graph['layer_edgeMat'][layer] + start_idx[i])
        edge_index = torch.cat(edge_mat_list, 1)
        return edge_index.to(self.args.device)

    def __process_sep_edgeIndex(self, batch_data, layer = 1):
        edge_mat_list = []
        start_pdx = [0]
        start_idx = [0]
        for i, graph in enumerate(batch_data['layer_data']):
            start_pdx.append(start_pdx[i] + graph['node_size'][layer - 1])
            start_idx.append(start_idx[i] + graph['node_size'][layer])
            edge_mat = torch.LongTensor(graph['interLayer_edgeMat'][layer]).clone()
            edge_mat[0] = edge_mat[0] + start_idx[i]
            edge_mat[1] = edge_mat[1] + start_pdx[i]
            edge_mat_list.append(edge_mat)
            '''
            edge_mat_list.append(
                torch.LongTensor(graph['interLayer_edgeMat'][layer]) +
                torch.LongTensor([start_idx[i], start_pdx[i]])
            )
            '''
        
        edge_index = torch.cat(edge_mat_list, dim = 1)
        return edge_index.to(self.args.device)

    def __process_sep_size(self, batch_data, layer=1):
        size = [
            (graph['node_size'][layer], graph['node_size'][layer - 1])
            for graph in batch_data['layer_data']
        ]
        return np.array(size).sum(axis=0)

    def __process_batch(self, batch_data, layer=0):
        batch = [
            [i] * graph['node_size'][layer]
            for i, graph in enumerate(batch_data['layer_data'])
        ]
        batch = reduce(lambda x, y: x + y, batch)
        return torch.tensor(batch, dtype=torch.long).to(self.args.device)

    def get_convs(self):
        convs = nn.ModuleList()
        _input_dim = self.num_features
        _output_dim = self.nhid
        for _ in range(self.args.tree_depth - 1):
            if self.args.conv == 'GCN':
                conv = GCNConv(_input_dim, _output_dim)
            elif self.args.conv == 'GAT':
                conv = GATConv(_input_dim,
                               _output_dim,
                               self.args.num_head,
                               concat=False)
            elif self.args.conv == 'Cheb':
                conv = ChebConv(_input_dim, _output_dim, K=2)
            elif self.args.conv == 'SAGE':
                conv = SAGEConv(_input_dim, _output_dim)
            elif self.args.conv == 'GAT2':
                conv = GATv2Conv(_input_dim,
                                 _output_dim,
                                 self.args.num_head,
                                 concat=False)
            elif self.args.conv == 'Transformer':
                conv = TransformerConv(_input_dim,
                                       _output_dim,
                                       2,
                                       concat=False)
            elif self.args.conv == 'GIN':
                conv = GINConv(nn.Sequential(
                    nn.Linear(_input_dim, _output_dim),
                    nn.ReLU(),
                    nn.Linear(_output_dim, _output_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(_output_dim),
                ),
                               train_eps=False)
            convs.append(conv)
            _input_dim = _output_dim
        return convs

    def get_sepool(self):
        pools = nn.ModuleList()
        _input_dim = self.nhid
        _output_dim = self.nhid
        for _ in range(self.args.tree_depth - 1):
            pool = SEPooling(
                nn.Sequential(
                    nn.Linear(_input_dim, _output_dim),
                    nn.ReLU(),
                    nn.Linear(_output_dim, _output_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(_output_dim),
                ))
            pools.append(pool)
            _input_dim = _output_dim
        return pools

    def get_classifier(self):
        init_dim = self.nhid * (self.args.tree_depth - 1)
        return nn.Sequential(
            nn.Linear(init_dim, self.nhid), nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid)
        )

    def forward(self, batch_data):
        x = torch.reshape(batch_data['x_list'], (-1, 768))
        
        x = self.feature_encoder(x)
        
        xs = []
        for _ in range(self.args.tree_depth - 1):
            # mp
            edge_index = self.__process_layer_edgeIndex(batch_data, _)
            x = F.relu(self.convs[_](x, edge_index))
            # sep
            if _ < self.args.tree_depth - 1:
                edge_index = self.__process_sep_edgeIndex(batch_data, _ + 1)
                size = self.__process_sep_size(batch_data, _ + 1)
                x = F.relu(self.sepools[_](x, edge_index, size=size))
            xs.append(x)

        pooled_xs = []
        for _, x in enumerate(xs):
            batch = self.__process_batch(batch_data, min(_ + 1, self.args.tree_depth - 1))
            pooled_x = self.global_pool(x, batch)
            pooled_xs.append(pooled_x)

        # For jumping knowledge scheme
        x = torch.cat(pooled_xs, dim=1)
        # For Classification
        x = self.classifier(x)
        return x