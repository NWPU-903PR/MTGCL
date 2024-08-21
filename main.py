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
from torch_geometric.utils import dropout_adj, dropout_adj, negative_sampling, remove_self_loops, add_self_loops


def train(data, train_mask, p_train, used_model):

    model.train()
    optimizer.zero_grad()

    if used_model == 'MTGCL':
        pre = model(data)
        output = pre[train_mask]
        Y = data.y[train_mask]
        gcl = model.sgcl_loss(data, train_mask, p_train)
        loss = F.binary_cross_entropy_with_logits(output, Y)
        loss = loss + args.λ * gcl
    elif used_model == 'EMOGI':
        pre = model(data)
        output = pre[train_mask]
        Y = data.y[train_mask]
        loss = F.binary_cross_entropy_with_logits(output, Y, pos_weight=torch.Tensor([45.0]).to(device))

    elif used_model == 'MTGCN':
        pre, r_loss, c1, c2 = model.MTGCN(data, args.dropout_edge, args.dropout, pb, E)
        output = pre[train_mask]
        Y = data.y[train_mask]
        loss = F.binary_cross_entropy_with_logits(output, Y) / (c1 * c1) + r_loss / (c2 * c2) + 2 * torch.log(c2 * c1)
    else:
        pre = model(data)
        output = pre[train_mask]
        Y = data.y[train_mask]
        loss = F.binary_cross_entropy_with_logits(output, Y)

    print(f'train loss {loss}')
    loss.backward()
    optimizer.step()


def test(data, test_mask, used_model):
    model.eval()
    with torch.no_grad():
        if used_model == 'MTCGN':
            pre, r_loss, c1, c2 = model.MTGCN(data, args.dropout_edge, args.dropout, pb, E)
        else:
            pre = model(data)
        output = pre[test_mask]
        Y = data.y[test_mask]
        auc, aupr, = quality_index(output, Y)
            #torch.cuda.empty_cache()
    return auc, aupr, pre

def quality_index(output, Y):

    pred = torch.sigmoid(output).cpu().detach().numpy()
    Yn = Y.cpu().numpy()

    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    aupr = metrics.auc(recall, precision)

    auc = metrics.roc_auc_score(Yn, pred)

    return auc, aupr


args = parse_args()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)


EPOCH = args.epochs
cv_num = args.cv_runs
num = args.num

AUC = np.zeros(shape=(num, cv_num))
AUPR = np.zeros(shape=(num, cv_num))

# ppi_name = ['CPDB', 'STRING', 'PathNet']
# ppi_type = 'STRING'

cancer = args.cancer
if cancer in ['BRCA', 'LIHC', 'COAD', 'PRAD', 'UCEC']:
    data = torch.load('./data/data_paper/CPDB_' + cancer + '_cancer.pkl', map_location=device)
else:
    data = torch.load('./data/data_paper/' + cancer + '_pan_cancer.pkl', map_location=device)
k_sets = data.k_sets

if args.model == 'MTGCN':
    pb, _ = remove_self_loops(data.edge_list)
    pb, _ = add_self_loops(pb)
    E = data.edge_list

time_start = time.time()
for i in range(num):
    root_dir = './result/'
    num_path = create_train_dir(root_dir, i)

    for cv_run in range(cv_num):
        print(cv_run)
        p_train, p_test, tr_mask, te_mask = k_sets[i][cv_run]
        cv_path = create_model_dir(num_path, cv_run)
        model = SGL(data, model_used=args.model, input_dim=data.x.shape[1], hidden_dim=args.hidden_dims[0], output_dim=args.hidden_dims[1],
                    drop_p=args.dropout, drop_edge_p=args.dropout_edge,
                    tau=args.tau, pe1=args.pe1, pe2=args.pe2, pf1=args.pf1, pf2=args.pf2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

        for epoch in range(0, EPOCH):
            print(epoch)
            train(data, tr_mask, p_train, args.model)

        torch.save(model, cv_path + 'model' + '_' + str(cv_run) + '.pkl')
        auc, aupr, x = test(data, te_mask, args.model)
        print(f' test auc {auc}, test aupr {aupr}')
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
