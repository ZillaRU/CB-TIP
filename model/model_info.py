from argparse import Namespace

import dgl
import torch.nn as nn

from model.losses_info import local_global_loss_

class FF_MI_max(nn.Module):
    def __init__(self, args: Namespace, gamma=.1):
        super(FF_MI_max, self).__init__()
        self.args = args
        self.gamma = gamma
        self.small_ff = FF_net(args.intra_out_dim, [96, 96], args.emb_dim)
        self.macro_ff = FF_net(args.intra_out_dim, [96, 96], args.emb_dim)
        self.drug_ff = FF_net(args.emb_dim, [128, 128], args.emb_dim)
        self.target_ff = FF_net(args.emb_dim, [128, 128], args.emb_dim)

    def forward(self, attrs, embs, pos_graph, neg_graph):
        measure = 'JSD'  # ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
        s_intra_attr = self.small_ff(attrs['small'])
        s_inter_enc = self.drug_ff(embs['drug'])[:s_intra_attr.shape[0], :]
        small_mol_local_global_loss = local_global_loss_(self.args, s_intra_attr, s_inter_enc,
                                                         dgl.to_homogeneous(dgl.node_subgraph(pos_graph, {
                                                             'drug': list(range(s_intra_attr.shape[0]))})),
                                                         dgl.to_homogeneous(dgl.node_subgraph(neg_graph, {
                                                             'drug': list(range(s_intra_attr.shape[0]))})),
                                                         measure)
        target_mol_local_global_loss = 0.
        if len(pos_graph.ntypes) > 1:
            t_intra_attr = self.macro_ff(attrs['target'])
            t_inter_enc = self.target_ff(embs['target'])
            assert t_intra_attr.shape[0] == embs['target'].shape[0]
            target_mol_local_global_loss = local_global_loss_(self.args, t_intra_attr, t_inter_enc,
                                                              dgl.to_homogeneous(dgl.node_subgraph(pos_graph, {
                                                                  'target': list(range(t_intra_attr.shape[0]))})),
                                                              None,
                                                              measure)
        return small_mol_local_global_loss + target_mol_local_global_loss


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