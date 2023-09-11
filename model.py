import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.autograd import Variable
from tqdm import tqdm
import dgl
import dgl.function as fn
import os

from LSTM import LSTMModel
from GCN import GCNModel
from Transform import TransformModel
from GAT import GAT
from GAT_dgl import GATModel
from GAT_dgl import Replace_GAT
from graph_feature_transfomr import GraphModel

from sklearn.svm import  LinearSVC



# class GCNLayer(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  out_feats,
#                  activation,
#                  dropout,
#                  layer_num=-1,
#                  bias=True):
#         super(GCNLayer, self).__init__()
#         self.layer_num = layer_num
#         self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_feats))
#         else:
#             self.bias = None
#         self.activation = activation
#         if dropout:
#             self.dropout = nn.Dropout(p=dropout)
#         else:
#             self.dropout = 0.
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, g, h):
#
#         # degs = g.in_degrees().float()
#         # norm = torch.pow(degs, -0.5)
#         # norm[torch.isinf(norm)] = 0
#         # g.ndata['norm'] = norm.unsqueeze(1)
#
#         if self.dropout:
#             h = self.dropout(h)
#         h = torch.mm(h, self.weight)
#         # normalization by square root of src degree
#         # h = h * g.ndata['norm']
#         g.ndata['h'] = h
#         g.update_all(fn.copy_src(src='h', out='m'),
#                           fn.sum(msg='m', out='h'))
#         h = g.ndata.pop('h')
#         # normalization by square root of dst degree
#         # h = h * g.ndata['norm']
#         # bias
#         if self.bias is not None:
#             h = h + self.bias
#         if self.activation:
#             h = self.activation(h)
#         if self.layer_num == 1:
#             h = dgl.mean_nodes(g, 'h')
#         return h
#
#
# class GCNLayer(nn.Module):
#     def __init__(self, in_feats, out_feats, activation,
#                  dropout,
#                  bias=True):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#
#
#         """multiply src with edge data or not"""
#         # self.msg_func = fn.copy_src(src='h', out='m')
#         self.msg_func = fn.src_mul_edge(src='h', edge='w', out='m')
#
#         self.reduce_func = fn.sum(msg='m', out='h')
#
#     def apply(self, nodes):
#         return {'h': F.relu(self.linear(nodes.data['h']))}
#
#     def forward(self, g, feature):
#         g.ndata['h'] = feature
#         g.update_all(self.msg_func, self.reduce_func)
#         g.apply_nodes(func=self.apply)
#         if self.last:
#             return dgl.mean_nodes(g, 'h')
#         else:
#             return g.ndata.pop('h')
#
#     def cat(self, g):
#         l = dgl.unbatch(g)
#         return torch.stack([g.ndata['h'].view(-1) for g in l], 0)
#
#     def max_pool(self, g):
#         l = dgl.unbatch(g)
#         return torch.stack([torch.max(g.ndata['h'], 0)[0] for g in l], 0)
#
#
# class GCNModel(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  n_hidden,
#                  n_classes,
#                  n_layers,
#                  activation,
#                  dropout):
#         super(GCNModel, self).__init__()
#         self.layers = nn.ModuleList()
#         # input layer
#         self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
#         # hidden layers
#         for i in range(n_layers - 1):
#             self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
#         # output layer
#         self.layers.append(GCNLayer(n_hidden, n_classes, None, dropout, 1))
#
#     def forward(self, g, features):
#         h = features
#         for layer in self.layers:
#             h = layer(g, h)
#
#         return h



class GCN_LSTM(nn.Module):
    def __init__(self,
                 gcn_in,
                 gcn_hid,
                 gcn_out,
                 gcn_layers,
                 lstm_hid,
                 lstm_out,
                 lstm_layers,
                 activation,
                 dropout):
        super(GCN_LSTM, self).__init__()

        self.GCN = GCNModel(gcn_in, gcn_hid, gcn_out, gcn_layers, activation, dropout=0.)
        self.LSTM = LSTMModel(gcn_out, lstm_hid, lstm_out, lstm_layers)

        self.dropout = nn.Dropout(dropout)

        # self.Linear = nn.Linear(output_dim, lstm_out)

    def forward(self, g1, features1, g2, features2, time_split):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        # device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        # gcn_out1 = self.GCN(g1, features1)
        # gcn_out2 = self.GCN(g2, features2)

        gcn_out1 = torch.stack([self.GCN(g1[i], features1[i]) for i in range(time_split)], 1)
        # print("gcn_out1 size", gcn_out1.size())
        gcn_out2 = torch.stack([self.GCN(g2[i], features2[i]) for i in range(time_split)], 1)
        # print("gcn_out2 size", gcn_out2.size())
        h, out = self.LSTM(gcn_out1, gcn_out2)
        out = self.dropout(out)
        # out = self.Linear(out)

        return out



class GCN_Transformer(nn.Module):
    def __init__(self,
                 gcn_in,
                 gcn_hid,
                 gcn_out,
                 gcn_layers,
                 activation,
                 time_length,
                 head_num,
                 transform_layer,
                 temporal_drop,
                 residual,
                 output_dim,
                 dropout):
        super(GCN_Transformer, self).__init__()

        self.GCN = GCNModel(gcn_in, gcn_hid, gcn_out, gcn_layers, activation, dropout=0.)
        self.GCN2 = GCNModel(gcn_in, gcn_hid, gcn_out, gcn_layers, activation, dropout=0.)

        self.Transform = TransformModel(gcn_out, time_length, head_num, transform_layer, temporal_drop, residual)
        # """transform_gcn"""
        # self.Transform = TransformModel(gcn_out, time_length, head_num, transform_layer, temporal_drop, residual)

        self.ln = nn.LayerNorm(gcn_out, elementwise_affine=True)

        """renji"""
        self.bn_1 = nn.BatchNorm1d(num_features=17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn_2 = nn.BatchNorm1d(num_features=17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        """data114 705"""
        # self.bn_1 = nn.BatchNorm1d(num_features=10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.bn_2 = nn.BatchNorm1d(num_features=10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.dropout = nn.Dropout(dropout)

        self.temporal_layer_config = list(map(int, transform_layer.split(",")))

        print("self.temporal_layer_config[-1]",self.temporal_layer_config[-1])
        print("output_dim", output_dim)
        self.Linear = nn.Linear(self.temporal_layer_config[-1], output_dim)

        # self.bn2 = nn.BatchNorm1d(num_features=self.temporal_layer_config[-1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, g1, features1, g2, features2, time_split):
        os.environ['CUDA_VISIBLE_DEVICES'] = '7'
        device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        # gcn_out1 = self.GCN(g1, features1)
        # gcn_out2 = self.GCN(g2, features2)

        # h1, a_grad_1, h_node_tensor_1 = self.GCN(g1[0], features1[0])
        # h2, a_grad_2, h_node_tensor_2 = self.GCN(g1[0], features1[0])


        grad_list = list()
        node_feature_list = list()

        gcn_out1_list = list()
        gcn_out2_list = list()


        """trans+GCN"""
        gcn_out1_list_without_readout = list()
        gcn_out2_list_without_readout = list()

        for i in range(time_split):
            h1, a_grad_1, h_node_tensor_1 = self.GCN(g1[i], features1[i])
            h2, a_grad_2, h_node_tensor_2 = self.GCN(g2[i], features2[i])
            gcn_out1_list.append(h1)
            gcn_out2_list.append(h2)
            # gcn_out1 = torch.stack([gcn_out1, h1],1)
            # gcn_out2 = torch.stack([gcn_out2, h2],1)

            gcn_out1_list_without_readout.append(h_node_tensor_1)
            gcn_out2_list_without_readout.append(h_node_tensor_2)



            if i==time_split-1:
                print("a_grad_1 len", len(a_grad_1))
                print("a_grad_1 size", a_grad_1[-1].size())
                print("h_node_tensor_1 size", h_node_tensor_1.size())


        """original"""
        # gcn_out1 = torch.stack(gcn_out1_list, 1)
        # gcn_out2 = torch.stack(gcn_out2_list, 1)
        #
        # # gcn_out1 = torch.stack([self.GCN(g1[i], features1[i]) for i in range(time_split)], 1)
        # # gcn_out2 = torch.stack([self.GCN2(g2[i], features2[i]) for i in range(time_split)], 1)
        #
        # # h1_end, a_grad_1, h_node_tensor_1 = self.GCN(g1[time_split-1], features1[time_split-1])
        # # h2_end, a_grad_2, h_node_tensor_2 = self.GCN(g1[time_split-1], features1[time_split-1])
        # #
        # # print("a_grad_1 len", len(a_grad_1))
        # # print("a_grad_1 size", a_grad_1[-1].size())
        # # print("h_node_tensor_1 size",h_node_tensor_1.size())
        #
        #
        # # h, out = self.LSTM(gcn_out1.to(device), gcn_out2.to(device))
        #
        # # print("gcn_out2", gcn_out2[22:25])
        #
        # print("gcn_out1.size", gcn_out1.size())     # [sample_num, time_split, feature]   [12,10,16]
        # print("gcn_out2.size", gcn_out2.size())
        #
        # gcn_out1 = self.bn_1(gcn_out1)
        # gcn_out2 = self.bn_2(gcn_out2)
        #
        # print("gcn_out1 after ln.size", gcn_out1.size())
        #
        # # gcn_out1 = self.ln(gcn_out1)
        #
        # # out = self.Transform(gcn_out1.to(device))
        # out, ST_dependecny = self.Transform(gcn_out1, gcn_out2)
        #
        #
        # #spatio-temporal dependency
        # # print("spatio-temporal dependency size", ST_dependecny.size())
        # # print(ST_dependecny)
        # # ST_dependecny
        #
        #
        # # print("transform out", out[22:25])
        #
        # # out = self.bn2(out)
        #
        # out = self.Linear(out)
        #
        # print("model_out size", out.size())           # [sample, class]  [12, 2]
        #
        # out = self.dropout(out)
        #
        # print("transform_out size", out.size())    # [sample, class]  [12, 2]





        """trans+GCN"""
        gcn_out1_without_readout = torch.stack(gcn_out1_list_without_readout, 1)
        gcn_out2_without_readout = torch.stack(gcn_out2_list_without_readout, 1)

        print("gcn_out1.size", gcn_out1_without_readout.size())     # [sample_num*node_num, time_split, feature]   [12*116,10,16]
        print("gcn_out2.size", gcn_out2_without_readout.size())

        gcn_out1 = self.bn_1(gcn_out1_without_readout)
        gcn_out2 = self.bn_2(gcn_out2_without_readout)

        print("gcn_out1 after ln.size", gcn_out1.size())


        out, ST_dependecny = self.Transform(gcn_out1, gcn_out2)

        out = self.Linear(out)

        """renji"""
        out = torch.reshape(out, (100, -1, 2))
        ST_dependecny = torch.reshape(ST_dependecny, (100, -1, 17, 17))

        # out = torch.reshape(out, (116, -1, 2))
        # ST_dependecny = torch.reshape(ST_dependecny, (116, -1, 10, 10))




        out = torch.mean(out, dim=0)

        print("model_out size", out.size())

        print("ST_dependecny size", ST_dependecny.size())

        ST_dependecny_final = torch.mean(ST_dependecny, dim=1)

        print("ST_dependecny_final size", ST_dependecny_final.size())

        # ST_dependecny = torch.reshape(ST_dependecny, (116, -1, 16))

        out = self.dropout(out)

        print("transform_out size", out.size())

        return out, ST_dependecny_final, a_grad_1[-1], a_grad_2[-1], h_node_tensor_1, h_node_tensor_2







class GAT_LSTM(nn.Module):
    def __init__(self,
                 gcn_in,
                 gcn_hid,
                 gcn_out,
                 gat_drop,
                 alpha,
                 gat_nheads,
                 gcn_layers,
                 lstm_hid,
                 lstm_out,
                 lstm_layers,
                 activation,
                 dropout):
        super(GAT_LSTM, self).__init__()

        self.GAT = GAT(gcn_in, gcn_hid, gcn_out, gat_drop, alpha, gat_nheads)
        # self.GAT2 = GAT(gcn_in, gcn_hid, gcn_out, gat_drop, alpha, gat_nheads)
        self.LSTM = LSTMModel(gcn_out, lstm_hid, lstm_out, lstm_layers)

        self.dropout = nn.Dropout(dropout)

        # self.Linear = nn.Linear(output_dim, lstm_out)

    def forward(self, g1, features1, g2, features2, time_split):
        os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        # gcn_out1 = self.GCN(g1, features1)
        # gcn_out2 = self.GCN(g2, features2)

        gat_out1 = torch.stack([self.GAT(g1[i], features1[i]) for i in range(time_split)], 1)
        # print("gcn_out1 size", gcn_out1.size())
        gat_out2 = torch.stack([self.GAT(g2[i], features2[i]) for i in range(time_split)], 1)
        # print("gcn_out2 size", gcn_out2.size())
        h, out = self.LSTM(gat_out1, gat_out2)
        out = self.dropout(out)
        # out = self.Linear(out)

        return out



class GAT_Transformer(nn.Module):
    def __init__(self,
                 gcn_in,
                 gcn_hid,
                 gcn_out,
                 gat_drop,
                 alpha,
                 gat_nheads,
                 time_length,
                 head_num,
                 transform_layer,
                 temporal_drop,
                 residual,
                 output_dim,
                 dropout):
        super(GAT_Transformer, self).__init__()

        self.GAT = GAT(gcn_in, gcn_hid, gcn_out, gat_drop, alpha, gat_nheads)
        self.GAT2 = GAT(gcn_in, gcn_hid, gcn_out, gat_drop, alpha, gat_nheads)
        self.Transform = TransformModel(gcn_out, time_length, head_num, transform_layer, temporal_drop, residual)

        self.ln = nn.LayerNorm(gcn_out, elementwise_affine=True)


        self.dropout = nn.Dropout(dropout)

        self.temporal_layer_config = list(map(int, transform_layer.split(",")))

        print("self.temporal_layer_config[-1]",self.temporal_layer_config[-1])
        print("output_dim", output_dim)



        # self.bn_1 = nn.BatchNorm1d(num_features=10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn_1 = nn.BatchNorm1d(num_features=17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        self.Linear = nn.Linear(self.temporal_layer_config[-1], output_dim)

        self.bn2 = nn.BatchNorm1d(num_features=self.temporal_layer_config[-1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, g1, features1, g2, features2, time_split):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        # device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        # gcn_out1 = self.GCN(g1, features1)
        # gcn_out2 = self.GCN(g2, features2)


        gat_out1 = torch.stack([self.GAT(g1[i], features1[i]) for i in range(time_split)], 1)
        gat_out2 = torch.stack([self.GAT2(g2[i], features2[i]) for i in range(time_split)], 1)
        # h, out = self.LSTM(gcn_out1.to(device), gcn_out2.to(device))

        # print("gcn_out2", gcn_out2[22:25])

        # print("gcn_out1.size", gcn_out1.size())
        gat_out1 = self.bn_1(gat_out1)
        gat_out2 = self.bn_2(gat_out2)

        # print("gcn_out1 after ln.size", gcn_out1.size())

        # gcn_out1 = self.ln(gcn_out1)

        # out = self.Transform(gcn_out1.to(device))
        out = self.Transform(gat_out1, gat_out2)

        # print("transform out", out[22:25])

        # out = self.bn2(out)

        out = self.Linear(out)

        print("model_out size", out.size())

        out = self.dropout(out)

        print("transform_out size", out.size())



        return out



class GAT_dgl_LSTM(nn.Module):
    def __init__(self,
                 node_num,
                 gat_num_layers,
                 gat_in_feats,
                 gat_hidden_feats,
                 gat_out_feats,
                 gat_num_heads,
                 gat_activation,
                 # gcn_in,
                 # gcn_hid,
                 gcn_out,
                 # gcn_layers,
                 lstm_hid,
                 lstm_out,
                 lstm_layers,
                 activation,
                 dropout):
        super(GAT_dgl_LSTM, self).__init__()

        # self.GCN = GCNModel(gcn_in, gcn_hid, gcn_out, gcn_layers, activation, dropout=0.)

        self.GAT = GATModel(node_num, gat_num_layers, gat_in_feats, gat_hidden_feats, gat_out_feats, gat_num_heads, gat_activation)


        self.LSTM = LSTMModel(gcn_out, lstm_hid, lstm_out, lstm_layers)

        self.dropout = nn.Dropout(dropout)

        # self.Linear = nn.Linear(output_dim, lstm_out)

    def forward(self, g1, features1, g2, features2, time_split):

        print("g1[0].device",g1[0].device)
        print("features1[0].device",features1[0].device)
        gcn_out1 = torch.stack([self.GAT(g1[i], features1[i]) for i in range(time_split)], 1)
        # print("gcn_out1 size", gcn_out1.size())
        # gcn_out2 = torch.stack([self.GAT(g2[i], features2[i]) for i in range(time_split)], 1)
        gcn_out2 = torch.stack([self.GAT(g2[i].add_self_loop(), features2[i]) for i in range(time_split)], 1)
        # print("gcn_out2 size", gcn_out2.size())
        h, out = self.LSTM(gcn_out1, gcn_out2)
        out = self.dropout(out)
        # out = self.Linear(out)

        return out




class GAT_dgl_Transformer(nn.Module):
    def __init__(self,
                 node_num,
                 gat_num_layers,
                 gat_in_feats,
                 gat_hidden_feats,
                 gat_out_feats,
                 gat_num_heads,
                 gat_activation,
                 gcn_out,
                 # gcn_layers,
                 # activation,
                 time_length,
                 head_num,
                 transform_layer,
                 temporal_drop,
                 residual,
                 output_dim,
                 dropout):
        super(GAT_dgl_Transformer, self).__init__()

        self.GAT = GATModel(node_num, gat_num_layers, gat_in_feats, gat_hidden_feats, gat_out_feats, gat_num_heads, gat_activation)
        self.GAT2 = GATModel(node_num, gat_num_layers, gat_in_feats, gat_hidden_feats, gat_out_feats, gat_num_heads, gat_activation)


        self.Transform = TransformModel(gcn_out, time_length, head_num, transform_layer, temporal_drop, residual)

        self.ln = nn.LayerNorm(gcn_out, elementwise_affine=True)
        # self.bn_1 = nn.BatchNorm1d(num_features=10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.bn_2 = nn.BatchNorm1d(num_features=10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.bn_1 = nn.BatchNorm1d(num_features=17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn_2 = nn.BatchNorm1d(num_features=17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        self.dropout = nn.Dropout(dropout)

        self.temporal_layer_config = list(map(int, transform_layer.split(",")))

        print("self.temporal_layer_config[-1]",self.temporal_layer_config[-1])
        print("output_dim", output_dim)
        self.Linear = nn.Linear(self.temporal_layer_config[-1], output_dim)

        # self.bn2 = nn.BatchNorm1d(num_features=self.temporal_layer_config[-1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, g1, features1, g2, features2, time_split):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        # gcn_out1 = self.GCN(g1, features1)
        # gcn_out2 = self.GCN(g2, features2)


        gcn_out1 = torch.stack([self.GAT(g1[i], features1[i]) for i in range(time_split)], 1)
        # gcn_out2 = torch.stack([self.GAT2(g2[i].add_self_loop(), features2[i]) for i in range(time_split)], 1)
        gcn_out2 = torch.stack([self.GAT2(g2[i], features2[i]) for i in range(time_split)], 1)


        # h, out = self.LSTM(gcn_out1.to(device), gcn_out2.to(device))

        # print("gcn_out2", gcn_out2[22:25])

        # print("gcn_out1.size", gcn_out1.size())
        gcn_out1 = self.bn_1(gcn_out1)
        gcn_out2 = self.bn_2(gcn_out2)

        # print("gcn_out1 after ln.size", gcn_out1.size())

        # gcn_out1 = self.ln(gcn_out1)

        # out = self.Transform(gcn_out1.to(device))
        out, ST_dependecny = self.Transform(gcn_out1, gcn_out2)

        # print("transform out", out[22:25])

        # out = self.bn2(out)

        out = self.Linear(out)

        # print("model_out size", out.size())

        out = self.dropout(out)

        # print("transform_out size", out.size())



        return out, ST_dependecny




class GCN_baseline(nn.Module):
    def __init__(self,
                 gcn_in,
                 gcn_hid,
                 gcn_out,
                 gcn_layers,
                 out_dim,
                 activation,
                 dropout,
                 time_length):
        super(GCN_baseline, self).__init__()

        self.GCN = GCNModel(gcn_in, gcn_hid, gcn_out, gcn_layers, activation, dropout=0.)

        self.dropout = nn.Dropout(dropout)

        self.Linear = nn.Linear(gcn_out*time_length, out_dim)

    def forward(self, g1, features1, g2, features2, time_split):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        # device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        # gcn_out1 = self.GCN(g1, features1)
        # gcn_out2 = self.GCN(g2, features2)

        gcn_out1 = torch.cat([self.GCN(g1[i], features1[i]) for i in range(time_split)], 1)
        print("gcn_out1 size", gcn_out1.size())


        out = self.dropout(gcn_out1)
        out = self.Linear(out)
        print("out size", out.size())

        return out




class LSTMbaseline(nn.Module):
    def __init__(self,
                 gcn_in,
                 gcn_hid,
                 gcn_out,
                 gcn_layers,
                 lstm_hid,
                 lstm_out,
                 lstm_layers,
                 activation,
                 dropout):
        super(LSTMbaseline, self).__init__()

        self.GCN = GraphModel(gcn_in, gcn_hid, gcn_out, gcn_layers, activation, dropout=0.)
        self.LSTM = LSTMModel(gcn_out, lstm_hid, lstm_out, lstm_layers)

        self.dropout = nn.Dropout(dropout)

        # self.Linear = nn.Linear(output_dim, lstm_out)

    def forward(self, g1, features1, g2, features2, time_split):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        # device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        # gcn_out1 = self.GCN(g1, features1)
        # gcn_out2 = self.GCN(g2, features2)

        gcn_out1 = torch.stack([self.GCN(g1[i], features1[i]) for i in range(time_split)], 1)
        print("gcn_out1 size", gcn_out1.size())
        gcn_out2 = torch.stack([self.GCN(g2[i], features2[i]) for i in range(time_split)], 1)
        # print("gcn_out2 size", gcn_out2.size())
        h, out = self.LSTM(gcn_out1, gcn_out2)
        out = self.dropout(out)
        # out = self.Linear(out)

        return out




class HDGCN(nn.Module):
    def __init__(self,
                 gcn_in,
                 gcn_hid,
                 gcn_out,
                 gcn_layers,
                 lstm_hid,
                 lstm_out,
                 lstm_layers,
                 activation,
                 dropout,
                 cluster,
                 output_dim):
        super(HDGCN, self).__init__()

        self.GCN = GCNModel(gcn_in, gcn_hid, gcn_out, gcn_layers, activation, dropout=0.)
        self.LSTM = LSTMModel(gcn_out, lstm_hid, lstm_out, lstm_layers)


        self.Assign_Matrix = nn.Parameter(torch.FloatTensor(size=(gcn_in, cluster)))


        self.dropout = nn.Dropout(dropout)

        self.Linear = nn.Linear(output_dim, lstm_out)

    def forward(self, g1, features1, g2, features2, time_split):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        # device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        # gcn_out1 = self.GCN(g1, features1)
        # gcn_out2 = self.GCN(g2, features2)

        gcn_out1 = torch.stack([self.GCN(g1[i], features1[i]) for i in range(time_split)], 1)
        # print("gcn_out1 size", gcn_out1.size())
        gcn_out2 = torch.stack([self.GCN(g2[i], features2[i]) for i in range(time_split)], 1)
        # print("gcn_out2 size", gcn_out2.size())
        h, out = self.LSTM(gcn_out1, gcn_out2)

        out = self.Assign_Matrix.T * out

        out = self.dropout(out)
        out = self.Linear(out)

        return out