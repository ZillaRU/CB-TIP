from argparse import Namespace

import torch

from .gan_losses import get_positive_expectation, get_negative_expectation


def local_global_loss_(args: Namespace, l_enc, g_enc, homo_rel_g_pos, homo_rel_g_neg, measure):
    if homo_rel_g_neg is None:
        num_nodes_pos = homo_rel_g_pos.num_nodes()
        node1s, node2s = homo_rel_g_pos.edges()
        node1s, node2s = node1s.long(), node2s.long()
        pos_adj = torch.zeros((num_nodes_pos, num_nodes_pos)) \
            .index_put_(indices=[node1s, node2s], values=torch.tensor(1.))
        pos_mask = pos_adj.to(args.device) + torch.eye(num_nodes_pos).to(args.device)
        neg_mask = torch.ones((num_nodes_pos, num_nodes_pos)).to(args.device) - pos_mask
    else:
        num_nodes_pos, num_nodes_neg = homo_rel_g_pos.num_nodes(), homo_rel_g_neg.num_nodes()
        assert num_nodes_pos == num_nodes_neg
        node1s, node2s = homo_rel_g_pos.edges()
        node1s, node2s = node1s.long(), node2s.long()
        pos_adj = torch.zeros((num_nodes_pos, num_nodes_pos)) \
            .index_put_(indices=[node1s, node2s], values=torch.tensor(1.))
        # values=torch.ones([homo_rel_g_pos.num_edges()]))
        node1s, node2s = homo_rel_g_neg.edges()
        node1s, node2s = node1s.long(), node2s.long()
        neg_adj = torch.zeros((num_nodes_pos, num_nodes_pos)) \
            .index_put_(indices=[node1s, node2s], values=torch.tensor(1.))
        # values=torch.ones([homo_rel_g_neg.num_edges()]))
        pos_mask = pos_adj.to(args.device) + torch.eye(num_nodes_pos).to(args.device)
        neg_mask = neg_adj.to(args.device)
        # neg_mask = torch.ones((num_nodes, num_nodes)).cuda() - pos_mask

    res = torch.mm(l_enc, g_enc.t()).to(args.device)
    num_edges = homo_rel_g_pos.num_edges() + num_nodes_pos
    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_edges
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes_pos ** 2 - 2 * num_edges)

    return (E_neg - E_pos)
