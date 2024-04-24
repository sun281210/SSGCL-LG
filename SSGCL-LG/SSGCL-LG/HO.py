import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import random
import tensorflow as tf
import pandas as pd
from dgl.nn.pytorch import edge_softmax, GATConv
import torch
#图注意力
class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel, weights=None):
        self.num_channel = num_channel
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1,1))
        nn.init.constant_(self.weight, 0.1)  # equal weight



    def forward(self, adj_list):
        adj_list = torch.stack(adj_list) # Row normalization of all graphs generated行
        adj_list = F.normalize(adj_list, dim=1, p=1)## Hadamard product + summation -> Conv

        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)



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

class HO(nn.Module):
    def __init__(self,in_dims,num_hidden,num_classes,num_layers,feat_drop,attn_drop
                 # g, in_dim_1, in_dim_2, hidden_dim, num_class,num_layer_1, heads, activation, f_drop, att_drop, slope, res

    ):
        super(HO, self).__init__()
        self.num_layers = num_layers
        self.hgat_layers = nn.ModuleList()
        activation = F.elu
        negative_slope=0.05
        self.activation = activation
        heads=[8] * num_layers + [1]

        self.dropout = nn.Dropout(p=feat_drop)

        #--------------------------------------这个是特征转换
        self.ntfc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for ntfc in self.ntfc_list:
            nn.init.xavier_normal_(ntfc.weight, gain=1.414)


        #--------------------------------------
        self.hgat_layers.append(GATConv(num_hidden, num_hidden, heads[0],feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            self.hgat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l],feat_drop, attn_drop, negative_slope, False, self.activation))
        # output projection
        self.hgat_layers.append(GATConv(num_hidden * heads[-2], num_hidden, heads[-1],feat_drop, attn_drop, negative_slope, False, None))

        self.gcnlayers = nn.ModuleList()
        self.gcnlayers.append(GraphConvolution(in_dims,num_hidden))
        for l in range(1, num_layers):
            self.gcnlayers.append(GraphConvolution(in_dims,num_hidden))

        self.lines=nn.Linear(num_hidden,num_classes,bias=True)
        nn.init.xavier_normal_(self.lines.weight, gain=1.414)
        self.predict = nn.Linear(num_hidden, num_classes)
        self.Graph=GraphChannelAttLayer(2)
    def forward(self, labeladj,f,mate,beta):
        # h = []  # 节点特征
        h2=[]#节点类型特征

        for ntfc, feature in zip(self.ntfc_list, f):
            h2.append(ntfc(feature))
        h = torch.cat(h2, 0)  # 逐行合并


        # #图注意力
        # adjM=self.Graph(mate)#55
        # labeladj=labeladj+adjM
        # g = F.normalize(labeladj, dim=1, p=2)  # 2范数归一化

        g = float(beta[0])* mate[0] + float(beta[1])* mate[1]
        g=labeladj+g
        g = F.normalize(g, dim=1, p=2)  # 2范数归一化



        for l in range(self.num_layers):
            h = self.gcnlayers[l](h, g).flatten(1)
        h = self.gcnlayers[-1](h, g)#.mean(1)

        h=h[:4278,:]

 #       z_mc = torch.cat(h2[0], 0)#逐行合并
        return h