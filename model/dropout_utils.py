#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch

from .norm_utils import _sparse_coo_tensor


def dropout_sparse(x, keep_prob):
    x = x.coalesce()
    i = x._indices()
    v = x._values()
    size = x.size()

    n = keep_prob + torch.rand(len(v))
    n = torch.floor(n).to(torch.bool)
    i = i[:, n]
    v = v[n]
    x = _sparse_coo_tensor(i, v, size=size)

    return x * (1. / keep_prob)


def dropout_dense(x, keep_prob):
    # print('dropout_dense()')
    x = x.clone()
    i = torch.nonzero(x, as_tuple=False)

    n = keep_prob + torch.rand(len(i))
    n = (1. - torch.floor(n)).to(torch.bool)
    x[i[n, 0], i[n, 1]] = 0.

    return x * (1. / keep_prob)


def dropout(x, keep_prob):
    if keep_prob == 1:
        return x
    if x.is_sparse:
        return dropout_sparse(x, keep_prob)
    else:
        return dropout_dense(x, keep_prob)
