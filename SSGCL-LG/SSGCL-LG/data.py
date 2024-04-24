import numpy as np
import torch.nn as nn
import scipy
import pickle
import torch
import dgl
import torch.nn.functional as F
import scipy.sparse as sp
import torch as th


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""#将scipy稀疏矩阵转换为torch稀疏张量
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def load_IMDB_data(prefix=r'F:\Desktop\SSGCL-LG\SSGCL-LG\IMDB'):



    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_0 = torch.FloatTensor(features_0)# 加载目标节点的特征
    features1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features1 = torch.FloatTensor(features1)
    features2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()  # 加载目标节点的特征
    features2 = torch.FloatTensor(features2)


    e = np.eye(3, dtype=np.float32)
    e = torch.FloatTensor(e)


    labels = np.load(prefix + '/labels.npy')  # 加载标签
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')  # 加载训练集，验证集，测试集的索引

    mam = np.load(prefix + '/mam_n.npy')  # 加载MAM的邻接矩阵
    mdm = np.load(prefix + '/mdm_n.npy')
    mam = torch.from_numpy(mam).type(torch.FloatTensor)
    mdm = torch.from_numpy(mdm).type(torch.FloatTensor)

    #构建M-A,M-D矩阵
    M_A = np.load(prefix + '/M_A.npy')
    M_D = np.load(prefix + '/M_D.npy')

    M_A = torch.from_numpy(M_A).type(torch.FloatTensor)
    M_D = torch.from_numpy(M_D).type(torch.FloatTensor)
    M_A = F.normalize(M_A, dim=1, p=2)
    M_D = F.normalize(M_D, dim=1, p=2)

    G = [M_A, M_D]  # 将两个矩阵存为列表形式


# 特征从numpy为tensor
    features=[features_0,features1,features2]
    feat = [features_0,e]

    labels = torch.LongTensor(labels)
    num_classes = 3

    train_idx = train_val_test_idx['train_idx']

    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']


    pos = sp.load_npz(prefix + '/pos_3.npz')
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    labeladj = np.load(prefix + '/labeladj_new.npy')
    labeladj = torch.from_numpy(labeladj).type(torch.FloatTensor)



    mate=[mam,mdm]

    return G, features,labeladj,feat,mate,pos, labels, num_classes, train_idx, val_idx, test_idx





