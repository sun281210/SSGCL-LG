import torch.nn as nn
import torch
import numpy as np
from HO import HO
from HE import HE
from contrast import Contrast

class SemanticAttention(nn.Module):  # 语义级注意力
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh(),nn.Linear(hidden_size, 1, bias=False))
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)  # 这里可以打印语义级注意力分配
        b = beta.expand((z.shape[0],) + beta.shape)
        return (b * z).sum(1)

class SSGCLLG(nn.Module):
    def __init__(self, num_meta_paths,in_size,in_dims, hidden_dim,out_size,num_layershe,num_layersho,dropout,tau,lam):
        super(SSGCLLG, self).__init__()
        self.he=HE( num_meta_paths, in_size, hidden_dim, out_size, num_layershe,dropout)
        self.ho=HO(in_dims,hidden_dim,out_size,num_layersho,dropout,#feat_drop
                                                             dropout,# tattn_drop
                     )
        self.contrast=Contrast(hidden_dim, tau, lam)
        self.predict = nn.Linear(hidden_dim, out_size)
        self.semantic_attention=SemanticAttention(hidden_dim,hidden_dim)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self,G, h ,labeladj,f,mate,pos):
        heembedings,beta = self.he(G, h)
        heembedings = self.dropout(heembedings)
        hoembedings = self.ho(labeladj, f,mate,beta)
        hoembedings = self.dropout(hoembedings)
        loss=self.contrast(heembedings,hoembedings,pos)

        return self.predict(heembedings),self.predict(hoembedings),loss,heembedings




