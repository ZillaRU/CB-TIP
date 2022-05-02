import numpy as np
import torch
import torch.nn as nn


class MultiInnerProductDecoder(nn.Module):
    def __init__(self, in_dim, num_et, params):
        super(MultiInnerProductDecoder, self).__init__()
        self.num_et = num_et #num_et - 2 for dd dt tt
        self.in_dim = in_dim
        self.weight = nn.Parameter(torch.Tensor(num_et, in_dim)).to(params.device)
        self.reset_parameters()

    def forward(self, graph, z, sigmoid=True):
        rel_types = graph.canonical_etypes
        res = {}
        for etype_id in range(len(rel_types)):
            edge_index0, edge_index1 = graph.edges(etype=rel_types[etype_id])
            value = (z[edge_index0.long()] * z[edge_index1.long()] * self.weight[etype_id]).sum(dim=1)
            # assert torch.equal(torch.isnan(value), torch.tensor(False, dtype=torch.bool))
            res[rel_types[etype_id]] = torch.sigmoid(value) if sigmoid else value
        return res

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))
