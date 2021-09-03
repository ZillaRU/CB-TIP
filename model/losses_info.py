from argparse import Namespace

import torch

from .gan_losses import get_positive_expectation, get_negative_expectation


def local_global_loss_(args: Namespace, l_enc, g_enc, homo_rel_g, measure):
    num_nodes = homo_rel_g.num_nodes()
    pos_adj = torch.zeros((num_nodes, num_nodes)) \
        .index_put_(homo_rel_g.edges(), value=torch.ones([homo_rel_g.num_edges()]))
    if args.cuda:
        pos_mask = pos_adj.cuda() + torch.eye(num_nodes).cuda()
        neg_mask = torch.ones((num_nodes, num_nodes)).cuda() - pos_mask
    else:
        pos_mask = pos_adj + torch.eye(num_nodes)
        neg_mask = torch.ones((num_nodes, num_nodes)) - pos_mask

    res = torch.mm(l_enc, g_enc.t())
    num_edges = homo_rel_g.num_edges() + num_nodes
    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_edges
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes ** 2 - 2 * num_edges)

    return (E_neg - E_pos)
