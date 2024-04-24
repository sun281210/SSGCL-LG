import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import random
import tensorflow as tf
import pandas as pd


class SemanticAttention(nn.Module):  # 语义级注意力
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size, 1, bias=False))
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)  # 这里可以打印语义级注意力分配
        b = beta.expand((z.shape[0],) + beta.shape)
        return (b * z).sum(1),beta


class GraphConvolution(nn.Module):  # 自己定义的GCN
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(out_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  # 这里的权重和偏置归一化
        #print(self.weight)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)# HW in GCN
        output = torch.spmm(adj, support) # AHW

        if self.bias is not None:
            return F.elu(output + self.bias)
        else:
            return F.elu(output)




class HANLayer(nn.Module):

    def __init__(self, num_meta_paths, in_size, out_size, dropout):
        super(HANLayer, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        for i in range(num_meta_paths):
            self.gcn_layers.append(GraphConvolution(in_size, out_size))  # 将原始的GAT改为了 GCN
        self.semantic_attention = SemanticAttention(in_size=out_size,hidden_size=out_size)
        self.num_meta_paths = num_meta_paths



    def get_0_1_array(self,array, rate=0.2):
        '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
        zeros_num = int(array.size * rate)  # 根据0的比率来得到 0的个数
        new_array = np.ones(array.size)  # 生成与原来模板相同的矩阵，全为1
        new_array[:zeros_num] = 0  # 将一部分换为0
        np.random.shuffle(new_array)  # 将0和1的顺序打乱
        re_array = new_array.reshape(array.shape)  # 重新定义矩阵的维度，与模板相同
        return re_array

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gcn_layers[i](h, g).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)


class HE(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_layers,dropout,):
        super(HE, self).__init__()
        self.num_meta_paths=num_meta_paths

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, hidden_size, bias=True) for in_dim in in_size])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, dropout))
        for l in range(1, num_layers):
            self.layers.append(HANLayer(num_meta_paths, hidden_size, hidden_size, dropout))
        self.predict = nn.Linear(hidden_size, out_size)

        self.semantic_attention = SemanticAttention(in_size=64, hidden_size=64)

    def forward(self, G, h):

        hz=[]#tezheng
        for fc, feature in zip(self.fc_list, h):
            hz.append(fc(feature))
        h = torch.cat(hz, 0)#逐行合并

        for gnn in self.layers:

            h,beta= gnn(G, h)
        h = h[:4278, :]

        return  h,beta