import argparse
import torch
from SSGCLLG import SSGCLLG
from tools import evaluate_results_nc
from pytorchtools import EarlyStopping
from data import load_IMDB_data

import numpy as np
import random
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
def main(args):
    G, features,labeladj,feat,mate, pos,labels, num_classes, train_idx, val_idx, test_idx = load_IMDB_data()

    features = [feature.to(args['device']) for feature in features]
    feats = [feat.to(args['device']) for feat in feat]

    labels = labels.to(args['device'])

    svm_macro_avg = np.zeros((7,), dtype=np.float)
    svm_micro_avg = np.zeros((7,), dtype=np.float)
    nmi_avg = 0
    ari_avg = 0
    print('start train with repeat = {}\n'.format(args['repeat']))
    for cur_repeat in range(args['repeat']):
        print('cur_repeat = {}   ==============================================================='.format(args['repeat']))
        model = SSGCLLG(num_meta_paths=len(G),
                     in_size=[feature.shape[1] for feature in features],
                     in_dims=[feat.shape[1] for feat in feats],
                     hidden_dim=args['hidden_units'],
                     out_size=num_classes,
                     num_layershe=args['num_layershe'],
                     num_layersho=args['num_layersho'],
                     dropout=args["dropout"],
                     tau=args["tau"],
                     lam=args["lam"])

        G = [graph.to(args['device']) for graph in G]



        early_stopping = EarlyStopping(patience=args['patience'], verbose=True,save_path='checkpoint/checkpoint_{}.pt'.format(args['dataset']))  # 提早停止，设置的耐心值为5
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])

        for epoch in range(args['num_epochs']):

            model.train()
            logits1,logits2,loss_self, h= model(G, features,labeladj,feat, mate,pos)
            loss1 = loss_fcn(logits1[train_idx], labels[train_idx])
            loss2 = loss_fcn(logits2[train_idx], labels[train_idx])
            loss = 0.8 * loss1 + 0.2 * loss2#0.8  0.2
            l = 0.5 # 0.3
            Loss = l * loss_self + (1 - l) * loss

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()


            model.eval()
            logits,_,loss_self, h = model(G, features,labeladj,feat,mate, pos)
            val_loss = loss_fcn(logits[val_idx], labels[val_idx])
            val_loss = l * loss_self + (1 - l) * val_loss
            test_loss = loss_fcn(logits[test_idx], labels[test_idx])
            test_loss = l*loss_self + (1-l)*test_loss
            print('Epoch{:d}| Train Loss{:.4f}| Val Loss{:.4f}| Test Loss{:.4f}'.format(epoch + 1, Loss.item(),val_loss.item(),test_loss.item()))
            early_stopping(val_loss.data.item(), model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        print('\ntesting...')
        model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(args['dataset'])))
        model.eval()
        logits,_,loss_self, h = model(G, features,labeladj,feat,mate, pos)
        svm_macro, svm_micro, nmi, ari = evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(),int(labels.max()) + 1)  # 使用SVM评估节点
        svm_macro_avg = svm_macro_avg + svm_macro
        svm_micro_avg = svm_micro_avg + svm_micro
    svm_macro_avg = svm_macro_avg / args['repeat']
    svm_micro_avg = svm_micro_avg / args['repeat']
    print('---\nThe average of {} results:'.format(args['repeat']))
    print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
    print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
    print('all finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='这是我们基于GAT所构建的HAN模型')
    parser.add_argument('--dataset', default='IMDB', help='数据集')
    parser.add_argument('--lr', default=0.005, help='学习率')#0.003
    parser.add_argument('--num_layershe', default=2, help='网络层数')#1
    parser.add_argument('--num_layersho', default=1, help='网络层数')#1
    parser.add_argument('--hidden_units', default=64, help='隐藏层维度')
    parser.add_argument('--dropout', default=0.1, help='丢弃率')#@0.1
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--weight_decay', default=0.001, help='权重衰减')#0.001
    parser.add_argument('--patience', type=int, default=7, help='耐心值')#2
    parser.add_argument('--seed', type=int, default=22,help='随机种子')#22
    parser.add_argument('--device', type=str, default='cpu:0', help='使用cuda:0或者cpu')
    parser.add_argument('--repeat', type=int, default=5, help='重复训练和测试次数')
    parser.add_argument('--tau', default=0.5)
    parser.add_argument('--lam', default=0.7)
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    main(args)

