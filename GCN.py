import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, last=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.last = last

        """multiply src with edge data or not"""
        # self.msg_func = fn.copy_src(src='h', out='m')
        self.msg_func = fn.src_mul_edge(src='h', edge='w', out='m')

        self.reduce_func = fn.sum(msg='m', out='h')

    def apply(self, nodes):
        # return {'h': F.relu(self.linear(nodes.data['h']))}

        return {'h': F.relu(self.linear(nodes.data['h']))}

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(self.msg_func, self.reduce_func)
        g.apply_nodes(func=self.apply)


        if self.last:
            return dgl.mean_nodes(g, 'h'), g.ndata.pop('h')
        else:
            return g.ndata.pop('h')

    def cat(self, g):
        l = dgl.unbatch(g)
        return torch.stack([g.ndata['h'].view(-1) for g in l], 0)

    def max_pool(self, g):
        l = dgl.unbatch(g)
        return torch.stack([torch.max(g.ndata['h'], 0)[0] for g in l], 0)


class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, gcn_layers, activation, dropout):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size, False)
        self.gcn2 = GCNLayer(hidden_size, num_classes, True)
        # self.linear = nn.Linear(hidden_size, num_classes)
        # self.dropout = nn.Dropout(0.5)

        self.bn = nn.BatchNorm1d(num_features=in_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.a_grad = list()

        # print("in_feats shape", in_feats)
        # print("hidden_size shape", hidden_size)
        # print("num_classes shape", num_classes)

    def grad_hook(self, grad):
        self.a_grad.append(grad)


    def forward(self, g, inputs):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        # inputs = self.bn(inputs)   #input size:  [sample_num*node_num, feature]

        # print("net", [layer.device for layer in self.gat_layers])
        h = self.gcn1(g, inputs)

        # print("gcn layer1", h)
        # print("gcn layer1 size", h.size())



        # weight = torch.reshape(h,(116,-1,32))
        # print("wight size", weight.size())
        #
        # weightmap = torch.mean(weight, dim=2)
        # print("weightmap size", weightmap.size())
        #
        # weight_vector = torch.mean(weightmap, dim=1)
        # print("weight_vector size", weight_vector.size())
        # print("weight_vector",weight_vector)

        h, h_node = self.gcn2(g, h)


        v = h_node.register_hook(self.grad_hook)

        if len(self.a_grad)==0:
            self.a_grad.append(h_node)
        # print("GCN model self.a_grad length", len(self.a_grad))


        # if len(self.a_grad)>1:
        #     print("a_grad length", len(self.a_grad))
        #
        #     print("grad size", self.a_grad[-1].size())
        #     h_node_tensor = torch.reshape(h_node, (116, -1, 16))
        #     print("h_node feature map size", h_node.size)
        #     print("node feature map size", h_node_tensor.size)



        # tensor_x =116
        # tensor_y = weight.size(1)
        # tensor_z = 16
        #
        # print("tensor_y", tensor_y)
        #
        # weight_matrix = torch.zeros([tensor_x,tensor_y,tensor_x])
        # print("weight_matrix init",weight_matrix.size())
        # for i in tqdm(range(tensor_y)):
        #     for j in range(tensor_x):
        #         for k in range(j, tensor_x):
        #             # print("torch.dot(weight[j,i,:], weight[k,i,:])", torch.dot(weight[j,i,:], weight[k,i,:]))
        #             # print("weight_matrix[j][i][k]",weight_matrix[j,i,k])
        #
        #             weight_matrix[j,i,k] = torch.dot(weight[j,i,:], weight[k,i,:])
        #             weight_matrix[k,i,j] = torch.dot(weight[j, i, :], weight[k, i, :])
        #
        # print("weight_matrix",weight_matrix)


        # print("gcn_out_hidden2", h[22:25])
        # h = self.linear(h)


        return h, self.a_grad, h_node
