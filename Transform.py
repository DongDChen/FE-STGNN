import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter

import copy




class TemporalAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 num_time_steps,
                 attn_drop,
                 residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs, EC):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs]  # [N, T, F]


        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        # k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(EC, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]

        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]



        outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        # diag_val = torch.ones_like(outputs[0])
        # tril = torch.tril(diag_val)
        # masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        # padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        # outputs = torch.where(masks == 0, padding, outputs)


        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs  # [h*N, T, T]


        # #spatio-temporal dependency
        # print("spatio-temporal dependency size", outputs.size())
        # print(outputs)



        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)


        ST_dependecny= outputs
        ST_dependecny = torch.cat(torch.split(ST_dependecny, split_size_or_sections=int(ST_dependecny.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]
        print("dropout attention map size", outputs.size())
        # print(outputs)
        # ST_dependecny1 = torch.cat(
        #     torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
        #     dim=2)
        # print("dropout ST_dependecny1 attention map size", ST_dependecny1.size())
        # print(ST_dependecny1)
        #
        # ST_dependecny2 = torch.cat(
        #     torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=2),
        #     dim=0)
        # print("dropout ST_dependecny2 attention map size", ST_dependecny2.size())
        # print(ST_dependecny2)



        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs

        #7: aggregation
        outputs = torch.mean(outputs, dim=1)

        return outputs, ST_dependecny

    def feedforward(self, inputs):
        # outputs = F.relu(self.lin(inputs))
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)



# gcn_out_fea=128, time_length=10 temporal_drop=0.5   value;
# head_num='16,8,8', transform_layer='128'   str;
# residual=True     bool;
class TransformModel(nn.Module):
    def __init__(self, gcn_out_fea, time_length, head_num, transform_layer, temporal_drop, residual):

        super(TransformModel, self).__init__()

        self.num_time_steps = time_length


        self.temporal_head_config = list(map(int, head_num.split(",")))
        self.temporal_layer_config = list(map(int, transform_layer.split(",")))

        self.temporal_drop = temporal_drop
        self.residual = residual

        # self.temporal_attn = self.build_model(gcn_out_fea)

        input_dim = gcn_out_fea


        self.layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[0],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.residual)

        # self.temporal_attention_layers = nn.Sequential()
        # print("self.temporal_layer_config", self.temporal_layer_config)
        # print("layer num", len(self.temporal_layer_config))
        # for i in range(len(self.temporal_layer_config)):
        #     self.layer = TemporalAttentionLayer(input_dim=input_dim,
        #                                    n_heads=self.temporal_head_config[i],
        #                                    num_time_steps=self.num_time_steps,
        #                                    attn_drop=self.temporal_drop,
        #                                    residual=self.residual)
        #     self.temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=self.layer)
        #     input_dim = self.temporal_layer_config[i]
        #
        #     print("input_dim", input_dim)
        #     print("n_heads", self.temporal_head_config[i])
        #     print("num_time_steps", self.num_time_steps)



    def forward(self, x, EC):

        # Temporal Attention forward
        # print("transformer input size", x.size())   #[N, Time, Feature]
        # temporal_out = self.temporal_attn(x, EC)
        temporal_out, ST_dependecny = self.layer(x, EC)

        return temporal_out, ST_dependecny


    # def build_model(self, gcn_out_fea):
    #
    #     input_dim = gcn_out_fea
    #
    #     temporal_attention_layers = nn.Sequential()
    #     print("self.temporal_layer_config", self.temporal_layer_config)
    #     print("layer num", len(self.temporal_layer_config))
    #     for i in range(len(self.temporal_layer_config)):
    #         layer = TemporalAttentionLayer(input_dim=input_dim,
    #                                        n_heads=self.temporal_head_config[i],
    #                                        num_time_steps=self.num_time_steps,
    #                                        attn_drop=self.temporal_drop,
    #                                        residual=self.residual)
    #         temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
    #         input_dim = self.temporal_layer_config[i]
    #
    #         print("input_dim", input_dim)
    #         print("n_heads", self.temporal_head_config[i])
    #         print("num_time_steps", self.num_time_steps)
    #
    #     return temporal_attention_layers
