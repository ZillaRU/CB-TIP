#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#

import numpy as np
import torch
#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch

from ..init_utils import init_glorot
from ..norm_utils import _sparse_coo_tensor


def dropout_sparse(x, keep_prob):
    x = x.coalesce()
    i = x._indices()
    v = x._values()
    size = x.size()

    n = keep_prob + torch.rand(len(v))
    n = torch.floor(n).to(torch.bool)
    i = i[:,n]
    v = v[n]
    x = _sparse_coo_tensor(i, v, size=size)

    return x * (1./keep_prob)


def dropout_dense(x, keep_prob):
    # print('dropout_dense()')
    x = x.clone()
    i = torch.nonzero(x, as_tuple=False)

    n = keep_prob + torch.rand(len(i))
    n = (1. - torch.floor(n)).to(torch.bool)
    x[i[n, 0], i[n, 1]] = 0.

    return x * (1./keep_prob)


def dropout(x, keep_prob):
    if keep_prob == 1:
        return x
    if x.is_sparse:
        return dropout_sparse(x, keep_prob)
    else:
        return dropout_dense(x, keep_prob)


class DEDICOMDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, drop_prob=0.,
        activation=torch.sigmoid, **kwargs):

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.drop_prob = drop_prob
        self.activation = activation

        self.global_interaction = init_glorot(input_dim, input_dim)
        self.local_variation = [
            torch.flatten(init_glorot(input_dim, 1)) \
                for _ in range(num_relation_types)
        ]

    def forward(self, inputs_row, inputs_col):
        outputs = []
        for k in range(self.num_relation_types):
            inputs_row = dropout(inputs_row, 1.-self.drop_prob)
            inputs_col = dropout(inputs_col, 1.-self.drop_prob)

            relation = torch.diag(self.local_variation[k])

            product1 = torch.mm(inputs_row, relation)
            product2 = torch.mm(product1, self.global_interaction)
            product3 = torch.mm(product2, relation)
            rec = torch.bmm(product3.view(product3.shape[0], 1, product3.shape[1]),
                inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
            rec = torch.flatten(rec)
            outputs.append(self.activation(rec))
        return outputs


class DistMultDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, drop_prob=0.,
        activation=torch.sigmoid, **kwargs):

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.drop_prob = drop_prob
        self.activation = activation

        self.relation = [
            torch.flatten(init_glorot(input_dim, 1)) \
                for _ in range(num_relation_types)
        ]

    def forward(self, inputs_row, inputs_col):
        outputs = []
        for k in range(self.num_relation_types):
            inputs_row = dropout(inputs_row, 1.-self.drop_prob)
            inputs_col = dropout(inputs_col, 1.-self.drop_prob)

            relation = torch.diag(self.relation[k])

            intermediate_product = torch.mm(inputs_row, relation)
            rec = torch.bmm(intermediate_product.view(intermediate_product.shape[0], 1, intermediate_product.shape[1]),
                inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
            rec = torch.flatten(rec)
            outputs.append(self.activation(rec))
        return outputs


class BilinearDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, drop_prob=0.,
        activation=torch.sigmoid, **kwargs):

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.drop_prob = drop_prob
        self.activation = activation

        self.relation = [
            init_glorot(input_dim, input_dim) \
                for _ in range(num_relation_types)
        ]

    def forward(self, inputs_row, inputs_col):
        outputs = []
        for k in range(self.num_relation_types):
            inputs_row = dropout(inputs_row, 1.-self.drop_prob)
            inputs_col = dropout(inputs_col, 1.-self.drop_prob)

            intermediate_product = torch.mm(inputs_row, self.relation[k])
            rec = torch.bmm(intermediate_product.view(intermediate_product.shape[0], 1, intermediate_product.shape[1]),
                inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
            rec = torch.flatten(rec)
            outputs.append(self.activation(rec))
        return outputs


class InnerProductDecoder(torch.nn.Module):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_relation_types, drop_prob=0.,
        activation=torch.sigmoid, **kwargs):

        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_relation_types = num_relation_types
        self.drop_prob = drop_prob
        self.activation = activation


    def forward(self, inputs_row, inputs_col):
        outputs = []
        for k in range(self.num_relation_types):
            inputs_row = dropout(inputs_row, 1.-self.drop_prob)
            inputs_col = dropout(inputs_col, 1.-self.drop_prob)

            rec = torch.bmm(inputs_row.view(inputs_row.shape[0], 1, inputs_row.shape[1]),
                inputs_col.view(inputs_col.shape[0], inputs_col.shape[1], 1))
            rec = torch.flatten(rec)
            outputs.append(self.activation(rec))
        return outputs
