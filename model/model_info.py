from argparse import Namespace

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.losses_info import local_global_loss_


# GcnInfomax1
# small drug的intra-inter对齐
# 忽略：macro-drug的intra-inter对齐
#   因为数量相对少，增加一组对齐要多很多参数。依靠small drug和target的对齐来调整intra-inter encoder的参数隐式的对齐，
#   对比：显式的对齐 GcnInfomax2
# target的intra-inter对齐
class GcnInfomax1(nn.Module):
    def __init__(self, args: Namespace, gamma=.1):
        super(GcnInfomax1, self).__init__()
        self.gamma = gamma
        self.prior = args.prior
        self.small_ff = FF_net(args.intra_out_dim, [96, 96], args.ff_intra)
        self.macro_ff = FF_net(args.intra_out_dim, [96, 96], args.ff_intra)
        self.drug_ff = FF_net(args.emb_dim, [128, 128], args.emb_dim)
        self.target_ff = FF_net(args.emb_dim, [128, 128], args.emb_dim)
        if self.prior:
            self.prior_d = PriorDiscriminator(args.emb_dim)

    def forward(self, attrs, embs, pos_graph, neg_graph, betas=[0.5, 0.5]):
        s_intra_attr = self.small_ff(attrs['small'])
        t_intra_attr = self.macro_ff(attrs['target'])
        s_inter_enc = self.drug_ff(embs['drug'])[:s_intra_attr.shape[0], :]
        t_inter_enc = self.target_ff(embs['target'])
        measure = 'JSD'  # ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
        # multi-relational ??? only in decoder ??
        assert t_intra_attr.shape[0] == embs['target'].shape[0]
        small_mol_local_global_loss = local_global_loss_(self.args, s_intra_attr, s_inter_enc,
                                                         dgl.to_homogeneous(dgl.node_subgraph(pos_graph, {'drug': list(range(s_intra_attr.shape[0]))})),
                                                         dgl.to_homogeneous(dgl.node_subgraph(neg_graph, {'drug': list(range(s_intra_attr.shape[0]))})),
                                                         measure)

        target_mol_local_global_loss = local_global_loss_(self.args, t_intra_attr, t_inter_enc,
                                                          dgl.to_homogeneous(dgl.node_subgraph(pos_graph, {'target': list(range(t_intra_attr.shape[0]))})),
                                                          dgl.to_homogeneous(dgl.node_subgraph(neg_graph, {'target': list(range(t_intra_attr.shape[0]))})),
                                                          measure)
        # small_target_local_global_loss = hete_local_global_loss_(self.args,
        #                                                          s_intra_attr, s_inter_enc,
        #                                                          t_intra_attr, t_inter_enc,
        #                                                          pos_graph['dt'], neg_graph['dt'],
        #                                                          measure)

        eps = 1e-5
        if self.prior:
            prior = torch.rand_like(s_inter_enc)
            term_a = torch.log(self.prior_d(prior) + eps).mean()
            term_b = torch.log(1.0 - self.prior_d(s_inter_enc) + eps).mean()
            PRIOR = - (term_a + term_b) * self.gamma
            prior = torch.rand_like(t_inter_enc)
            term_a = torch.log(self.prior_d(prior) + eps).mean()
            term_b = torch.log(1.0 - self.prior_d(t_inter_enc) + eps).mean()
            PRIOR -= (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return small_mol_local_global_loss + target_mol_local_global_loss + PRIOR
        # local_global_loss + PRIOR


# class GcnInfomax2(nn.Module):
#     def __init__(self, args: Namespace, gamma=.1):
#         super(GcnInfomax2, self).__init__()
#         self.args = args
#         self.gamma = gamma
#         self.prior = args.prior
#         self.features_dim = args.hidden_size
#         self.embedding_dim = args.gcn_hidden3
#         self.small_ff = FF_local(args, self.intra_out_dim)
#         self.macro_ff = FF_local(args, self.intra_out_dim)
#         self.drug_ff = FF_global(args, self.emb_dim)
#         self.target_ff = FF_global(args, self.emb_dim)
#
#         if self.prior:
#             self.prior_d = PriorDiscriminator(self.embedding_dim)
#
#     def forward(self, attrs, embs, pos_graph, neg_graph):
#         g_enc = self.global_d(embeddings)
#         l_enc = self.local_d(features)
#         measure = 'JSD'  # ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
#         local_global_loss = local_global_loss_(self.args, l_enc, g_enc, graph, num_drugs, measure)
#         eps = 1e-5
#         if self.prior:
#             prior = torch.rand_like(embeddings)
#             term_a = torch.log(self.prior_d(prior) + eps).mean()
#             term_b = torch.log(1.0 - self.prior_d(embeddings) + eps).mean()
#             PRIOR = - (term_a + term_b) * self.gamma
#         else:
#             PRIOR = 0
#
#         return local_global_loss + PRIOR


# class GlobalDiscriminator(nn.Module):
#     def __init__(self, args, input_dim):
#         super().__init__()
#
#         self.l0 = nn.Linear(32, 32)
#         self.l1 = nn.Linear(32, 32)
#         self.l2 = nn.Linear(512, 1)
#
#     def forward(self, y, M, data):
#         adj = Variable(data['adj'].float(), requires_grad=False).cuda()
#         batch_num_nodes = data['num_nodes'].int().numpy()
#         M, _ = self.encoder(M, adj, batch_num_nodes)
#         h = torch.cat((y, M), dim=1)
#         h = F.relu(self.l0(h))
#         h = F.relu(self.l1(h))
#         return self.l2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class FF_net(nn.Module):
    def __init__(self, _in, hiddens: list, _out):
        super(FF_net, self).__init__()
        self.block = nn.Sequential()
        hiddens.insert(0, _in)
        hiddens.append(_out)
        for i in range(1, len(hiddens)):
            self.block.add_module(f'linear{hiddens[i - 1]}-{hiddens[i]}',
                                  nn.Linear(hiddens[i - 1], hiddens[i]))
            self.block.add_module('relu', nn.ReLU())
        self.linear_shortcut = nn.Linear(_in, _out)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

    # def reset_parameters(self): # refer to nn.Linear source code
    #     pass init.kaiming_uniform_


class FF_net3(nn.Module):
    def __init__(self, _in, hiddens: list, _out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(_in, hiddens[0]),
            nn.ReLU(),
            nn.Linear(hiddens[0], hiddens[1]),
            nn.ReLU(),
            nn.Linear(hiddens[1], _out),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(_in, _out)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class FF_net2(nn.Module):
    def __init__(self, _in, hidden, _out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, _out),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(_in, _out)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class FF_net1(nn.Module):
    def __init__(self, input_dim_1, input_dim_2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim_1, input_dim_2),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim_2, input_dim_2)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
