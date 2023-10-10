import numpy as np
import pandas as pd
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch_geometric.utils import dropout_adj
import torch.nn as nn
import math
from torch_geometric.typing import Adj, OptTensor, PairTensor

from math import sqrt
from torch_geometric.utils import degree, to_undirected
from torch.autograd import Variable


class SGL(torch.nn.Module):
    def __init__(self, data, input_dim: int, hidden_dim: int, output_dim: int, drop_p: float = 0.5, drop_edge_p: float = 0.5, tau: float = 0.5, pe1: float = 0.5, pe2: float = 0.5,
                 pf1: float = 0., pf2: float = 0.):

        super(SGL, self).__init__()

        self.drop_p = drop_p
        self.drop_edge_p = drop_edge_p
        self.tau = tau
        self.pe1 = pe1
        self.pe2 = pe2
        self.pf1 = pf1
        self.pf2 = pf2

        #GAT
        # self.add_self_loops = True
        # self.conv1 = GATConv(input_dim, 300, heads=3, concat=False, negative_slope=0.01, dropout=0.2, add_self_loops = self.add_self_loops)
        # self.conv2 = GATConv(300, 100, heads=3, concat=False, negative_slope=0.01, dropout=0.2,
        #                     add_self_loops= self.add_self_loops)
        # self.conv3 = GATConv(100, 1, heads=3, concat=False, negative_slope=0.01, dropout=0.2,
        #                      add_self_loops= self.add_self_loops)

        # ChebConv
        # self.conv1 = ChebConv(input_dim, hidden_dim, K=2, normalization='sym')
        # self.conv2 = ChebConv(hidden_dim, output_dim, K=2, normalization='sym')
        # self.conv3 = ChebConv(output_dim, 1, K=2, normalization='sym')

        # GCN
        self.add_self_loops = False
        self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=self.add_self_loops)
        self.conv2 = GCNConv(2*hidden_dim, output_dim, add_self_loops=self.add_self_loops)
        self.conv3 = GCNConv(output_dim, 1, add_self_loops=self.add_self_loops)

        self.fc = torch.nn.Linear(input_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(2*hidden_dim, output_dim)
        self.fc2 = torch.nn.Linear(output_dim, 1)

        self.fc_output = torch.nn.Linear(2 * output_dim, 1)

        self.fc3 = torch.nn.Linear(output_dim, 2 * output_dim)
        self.fc4 = torch.nn.Linear(2 * output_dim, output_dim)


    def forward(self, data, edge_weight: OptTensor = None):

        pre, _ = self.gcn_mlp(data, self.drop_p, self.drop_edge_p)

        return pre
    def sgcl_loss(self, data, train_mask, p_train):
        _, z1 = self.gcn_mlp(data, self.pe1, self.pf1)

        _, z2 = self.gcn_mlp(data, self.pe2, self.pf2)
        z1 = torch.relu(self.fc3(z1))
        z1 = self.fc4(z1)

        z2 = torch.relu(self.fc3(z2))
        z2 = self.fc4(z2)

        l1 = self.semi_loss_sgcl(z1, z2, data, train_mask, p_train)
        l2 = self.semi_loss_sgcl(z2, z1, data, train_mask, p_train)

        l = (l1 + l2) * 0.5
        l = l.mean()
        return l
    def gcn_mlp(self, data, p_edge, p_features):

        edge_index, _ = dropout_adj(data.edge_list, p=p_edge,
                                        force_undirected=True,
                                        num_nodes=data.x.shape[0],
                                        training=self.training)

        x = F.dropout(data.x, p=p_features, training=self.training)

        x1 = torch.relu(self.fc(x))
        x2 = torch.relu(self.conv1(x, edge_index))
        x3 = torch.cat((x1, x2), 1)
        x4 = F.dropout(x3, p=p_features, training=self.training)

        x5 = torch.relu(self.fc1(x4))
        x6 = torch.relu(self.conv2(x4, edge_index))
        x7 = x5 + x6
        x8 = F.dropout(x7, p=p_features, training=self.training)

        x9 = self.fc2(x8)
        x10 = self.conv3(x8, edge_index)
        pre = x9 + x10

        return pre, x7

    def mlp(self,  data, p_edge, p_features):

        x = F.dropout(data.x, p=p_features, training=self.training)
        x_1 = torch.relu(self.fc(x))
        x_1 = F.dropout(x_1, p=p_features, training=self.training)

        x_1 = torch.relu(self.fc1(x_1))
        x_2 = F.dropout(x_1, p=p_features, training=self.training)

        pre = self.fc2(x_2)

        return pre, 0

    def gcn(self,  data, p_edge, p_features):
        #gcn
        edge_index, _ = dropout_adj(data.edge_list, p=p_edge,
                                    force_undirected=True,
                                    num_nodes=data.x.shape[0],
                                    training=self.training)

        x = F.dropout(data.x, p=p_features, training=self.training)
        x_1 = torch.relu(self.conv1(x, edge_index))
        x_1 = F.dropout(x_1, p=p_features, training=self.training)
        x_2 = torch.relu(self.conv2(x_1, edge_index))
        x_3 = F.dropout(x_2, p=p_features, training=self.training)
        pre = self.conv3(x_3, edge_index)
        return pre, x_2

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss_sgcl(self, z1, z2, data, train_mask, p_train):
        f = lambda x: torch.exp(x / self.tau)
        ff = lambda x: x / self.tau
        p_train_index = torch.nonzero(torch.FloatTensor(p_train))[:, 0]
        f_train_index = torch.nonzero(torch.FloatTensor(np.logical_xor(train_mask, p_train.T)))[:, 0]
        rall_train_index = torch.nonzero(~torch.FloatTensor(train_mask).to(bool))[:, 0]

        a = self.sim(z1[p_train_index], z2[p_train_index])
        b = self.sim(z1[p_train_index], z1[p_train_index])

        c = self.sim(z1[f_train_index], z2[f_train_index])
        d = self.sim(z1[f_train_index], z1[f_train_index])

        p_smi = 1 / (2 * len(p_train_index) - 1) * (ff(a).sum(1) + ff(b).sum(1)- ff(b).diag()) \
                - torch.log(f(self.sim(z1[p_train_index], z1)).sum(1) + f(self.sim(z1[p_train_index], z2)).sum(1) - f(b).diag())

        f_smi = 1 / (2 * len(f_train_index) - 1) * (ff(c).sum(1) + ff(d).sum(1) - ff(d).diag())\
                - torch.log(f(self.sim(z1[f_train_index], z1)).sum(1) + f(self.sim(z1[f_train_index], z2)).sum(1) - f(d).diag())

        npf_smi = f(self.sim(z1[rall_train_index], z2[rall_train_index])).diag()
        npf_smi_1 = f(self.sim(z1[rall_train_index], z2)).sum(1) + f(self.sim(z1[rall_train_index], z1)).sum(1) - f(
            self.sim(z1[rall_train_index], z1[rall_train_index])).diag()
        return -(torch.cat((torch.cat((p_smi, f_smi), 0), torch.log(npf_smi / npf_smi_1)), 0))
