import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import time
from model_SGL import SGL
from sklearn import metrics
import pickle
import os
from utils import parse_args, create_train_dir, create_model_dir, write_hyper_params
import torch.nn as nn

def train(data, train_mask, p_train):

    model.train()
    optimizer.zero_grad()
    pre = model(data)

    loss, auc, aupr = quality_index(pre[train_mask], data.y[train_mask])
    rl = model.sgcl_loss(data, train_mask, p_train)
    #rl = 0
    loss = loss + args.λ * rl
    print(f'train loss {loss}, train auc={auc}, train au_pr={aupr}')
    loss.backward()
    optimizer.step()


def test(data, test_mask, p_test):
    model.eval()
    pre = model(data)
    #rl = model.sgcl_loss(data, test_mask, p_test)
    loss, auc, aupr, = quality_index(pre[test_mask], data.y[test_mask])

    rl=0
    return loss+args.λ*rl, auc, aupr, pre

def quality_index(output, Y):
    #loss = F.binary_cross_entropy_with_logits(output, Y, pos_weight=torch.Tensor([45.0]).to(device))
    loss = F.binary_cross_entropy_with_logits(output, Y)
    pred = torch.sigmoid(output).cpu().detach().numpy()
    Yn = Y.cpu().numpy()

    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    aupr = metrics.auc(recall, precision)

    auc = metrics.roc_auc_score(Yn, pred)

    return loss, auc, aupr


args = parse_args()
device = torch.device(args.device)
seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)


EPOCH = args.epochs
cv_num = args.cv_runs
num = 10


AUC = np.zeros(shape=(num, cv_num))
AUPR = np.zeros(shape=(num, cv_num))


cancer = args.cancer
if cancer in ['BRCA', 'LIHC']:
    data = torch.load('./data/data_paper/CPDB_' + cancer + '_cancer.pkl', map_location=device)
else:
    data = torch.load('./data/data_paper/' + cancer + '_pan_cancer.pkl', map_location=device)
k_sets = data.k_sets
time_start = time.time()
for i in range(num):
    root_dir = './result/'
    num_path = create_train_dir(root_dir, i)

    for cv_run in range(cv_num):
        print(cv_run)
        p_train, p_test, tr_mask, te_mask = k_sets[i][cv_run]
        cv_path = create_model_dir(num_path, cv_run)
        model = SGL(data, input_dim=data.x.shape[1], hidden_dim=args.hidden_dims[0], output_dim=args.hidden_dims[1],
                    drop_p=args.dropout, drop_edge_p=args.dropout_edge,
                    tau=args.tau, pe1=args.pe1, pe2=args.pe2, pf1=args.pf1, pf2=args.pf2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

        for epoch in range(0, EPOCH):
            print(epoch)
            train(data, tr_mask, p_train)
            if (epoch+1) % 2 == 0:
                test_loss, auc, aupr, x = test(data, te_mask, p_test)
                print(f' test auc {auc}, test aupr {aupr}')


        torch.save(model, cv_path + 'model' + '_' + str(cv_run) + '.pkl')
        test_loss, auc, aupr, x = test(data, te_mask, p_test)
        AUC[i][cv_run] = auc
        AUPR[i][cv_run] = aupr
        pred = torch.sigmoid(x).cpu().detach().numpy()
        pd.DataFrame(pred).to_csv(cv_path + 'predict' + '_' + str(cv_run) + '.txt')

print(time.time() - time_start)
print(AUC.mean())
print(AUC.mean(1).std())
print(AUPR.mean())
print(AUPR.mean(1).std())

with open('result/auc.pkl', 'wb') as fo:
    pickle.dump(AUC, fo)
with open('result/au_pr.pkl', 'wb') as fo:
    pickle.dump(AUPR, fo)

write_hyper_params(vars(args), os.path.join('./result/', 'hyper_params.txt'))
