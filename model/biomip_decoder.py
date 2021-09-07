import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################################################
# Decoder - Multirelational Link Prediction
#########################################################################
from dgl._deprecate.graph import DGLGraph

from model.dropout_utils import dropout
from model.init_utils import init_glorot


class MultiInnerProductDecoder(nn.Module):
    def __init__(self, in_dim, num_et):
        super(MultiInnerProductDecoder, self).__init__()
        self.num_et = num_et-2
        self.in_dim = in_dim
        self.weight = nn.Parameter(torch.Tensor(num_et, in_dim))
        self.reset_parameters()

    # def forward(self, z, edge_index, edge_type, sigmoid=True):
        # value = (z[edge_index[0]] * z[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        # return torch.sigmoid(value) if sigmoid else value
    def forward(self,graph, z, sigmoid=True, test=False):
        # if test: # return res of each type in dict
        # if isinstance(graph, DGLGraph):
        rel_types = graph.canonical_etypes[:-2]
        res = {}
        for etype_id in range(len(rel_types)):
            edge_index0, edge_index1 = graph.edges(etype=rel_types[etype_id])
            value = (z[edge_index0.long()] * z[edge_index1.long()] * self.weight[etype_id]).sum(dim=1)
            # assert torch.equal(torch.isnan(value), torch.tensor(False, dtype=torch.bool))
            res[rel_types[etype_id]] = torch.sigmoid(value) if sigmoid else value
            # else:
        #     raise NotImplementedError
        return res

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))


class DEDICOMDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, keep_prob=1.,
                 activation=torch.sigmoid, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.keep_prob = keep_prob
        self.activation = activation

        self.global_interaction = torch.nn.Parameter(init_glorot(input_dim, input_dim))
        self.local_variation = torch.nn.ParameterList([
            torch.nn.Parameter(torch.flatten(init_glorot(input_dim, 1))) \
            for _ in range(num_relation_types)
        ])

    def forward(self, inputs_row, inputs_col, relation_index):
        inputs_row = dropout(inputs_row, self.keep_prob)
        inputs_col = dropout(inputs_col, self.keep_prob)

        relation = torch.diag(self.local_variation[relation_index])

        product1 = torch.mm(inputs_row, relation)
        product2 = torch.mm(product1, self.global_interaction)
        product3 = torch.mm(product2, relation)
        rec = torch.bmm(product3.view(product3.shape[0], 1, product3.shape[1]),
                        inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
        rec = torch.flatten(rec)

        return self.activation(rec)


class DistMultDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, keep_prob=1.,
                 activation=torch.sigmoid, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.keep_prob = keep_prob
        self.activation = activation

        self.relation = torch.nn.ParameterList([
            torch.nn.Parameter(torch.flatten(init_glorot(input_dim, 1))) \
            for _ in range(num_relation_types)
        ])

    def forward(self, inputs_row, inputs_col, relation_index):
        inputs_row = dropout(inputs_row, self.keep_prob)
        inputs_col = dropout(inputs_col, self.keep_prob)

        relation = torch.diag(self.relation[relation_index])

        intermediate_product = torch.mm(inputs_row, relation)
        rec = torch.bmm(intermediate_product.view(intermediate_product.shape[0], 1, intermediate_product.shape[1]),
                        inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
        rec = torch.flatten(rec)

        return self.activation(rec)


class BilinearDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, keep_prob=1.,
                 activation=torch.sigmoid, **kwargs):

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.keep_prob = keep_prob
        self.activation = activation

        self.relation = torch.nn.ParameterList([
            torch.nn.Parameter(init_glorot(input_dim, input_dim)) \
            for _ in range(num_relation_types)
        ])

    def forward(self, inputs_row, inputs_col, relation_index):
        inputs_row = dropout(inputs_row, self.keep_prob)
        inputs_col = dropout(inputs_col, self.keep_prob)

        intermediate_product = torch.mm(inputs_row, self.relation[relation_index])
        rec = torch.bmm(intermediate_product.view(intermediate_product.shape[0], 1, intermediate_product.shape[1]),
                        inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
        rec = torch.flatten(rec)

        return self.activation(rec)


class NNDecoder(torch.nn.Module):
    def __init__(self, in_dim, num_uni_edge_type, l1_dim=16):
        """ in_dim: the feat dim of a drug
            num_edge_type: num of dd edge type """

        super(NNDecoder, self).__init__()
        self.l1_dim = l1_dim  # Decoder Lays' dim setting

        # parameters
        # for drug 1
        self.w1_l1 = nn.Parameter(torch.Tensor(in_dim, l1_dim))
        self.w1_l2 = nn.Parameter(torch.Tensor(num_uni_edge_type, l1_dim))  # dd_et
        # specified
        # for drug 2
        self.w2_l1 = nn.Parameter(torch.Tensor(in_dim, l1_dim))
        self.w2_l2 = nn.Parameter(torch.Tensor(num_uni_edge_type, l1_dim))  # dd_et
        # specified

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type):
        # layer 1
        d1 = torch.matmul(z[edge_index[0]], self.w1_l1)
        d2 = torch.matmul(z[edge_index[1]], self.w2_l1)
        d1 = F.relu(d1, inplace=True)
        d2 = F.relu(d2, inplace=True)

        # layer 2
        d1 = (d1 * self.w1_l2[edge_type]).sum(dim=1)
        d2 = (d2 * self.w2_l2[edge_type]).sum(dim=1)

        return torch.sigmoid(d1 + d2)

    def reset_parameters(self):
        self.w1_l1.data.normal_()
        self.w2_l1.data.normal_()
        self.w1_l2.data.normal_(std=1 / np.sqrt(self.l1_dim))
        self.w2_l2.data.normal_(std=1 / np.sqrt(self.l1_dim))

