#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import numpy as np
import scipy.sparse as sp
import torch


def _check_tensor(adj_mat):
    if not isinstance(adj_mat, torch.Tensor):
        raise ValueError('adj_mat must be a torch.Tensor')


def _check_sparse(adj_mat):
    if not adj_mat.is_sparse:
        raise ValueError('adj_mat must be sparse')


def _check_dense(adj_mat):
    if adj_mat.is_sparse:
        raise ValueError('adj_mat must be dense')


def _check_square(adj_mat):
    if len(adj_mat.shape) != 2 or \
            adj_mat.shape[0] != adj_mat.shape[1]:
        raise ValueError('adj_mat must be a square matrix')


def _check_2d(adj_mat):
    if len(adj_mat.shape) != 2:
        raise ValueError('adj_mat must be a square matrix')


def _sparse_coo_tensor(indices, values, size):
    ctor = { torch.float32: torch.sparse.FloatTensor,
             torch.float32: torch.sparse.DoubleTensor,
             torch.uint8: torch.sparse.ByteTensor,
             torch.long: torch.sparse.LongTensor,
             torch.int: torch.sparse.IntTensor,
             torch.short: torch.sparse.ShortTensor,
             torch.bool: torch.sparse.ByteTensor }[values.dtype]
    return ctor(indices, values, size)


def add_eye_sparse(adj_mat: torch.Tensor) -> torch.Tensor:
    _check_tensor(adj_mat)
    _check_sparse(adj_mat)
    _check_square(adj_mat)

    adj_mat = adj_mat.coalesce()
    indices = adj_mat.indices()
    values = adj_mat.values()

    eye_indices = torch.arange(adj_mat.shape[0], dtype=indices.dtype,
                               device=adj_mat.device).view(1, -1)
    eye_indices = torch.cat((eye_indices, eye_indices), 0)
    eye_values = torch.ones(adj_mat.shape[0], dtype=values.dtype,
                            device=adj_mat.device)

    indices = torch.cat((indices, eye_indices), 1)
    values = torch.cat((values, eye_values), 0)

    adj_mat = _sparse_coo_tensor(indices, values, adj_mat.shape)

    return adj_mat


def norm_adj_mat_one_node_type_sparse(adj_mat: torch.Tensor) -> torch.Tensor:
    _check_tensor(adj_mat)
    _check_sparse(adj_mat)
    _check_square(adj_mat)

    adj_mat = add_eye_sparse(adj_mat)
    adj_mat = norm_adj_mat_two_node_types_sparse(adj_mat)

    return adj_mat


def norm_adj_mat_one_node_type_dense(adj_mat: torch.Tensor) -> torch.Tensor:
    _check_tensor(adj_mat)
    _check_dense(adj_mat)
    _check_square(adj_mat)

    adj_mat = adj_mat + torch.eye(adj_mat.shape[0], dtype=adj_mat.dtype,
                                  device=adj_mat.device)
    adj_mat = norm_adj_mat_two_node_types_dense(adj_mat)

    return adj_mat


def norm_adj_mat_one_node_type(adj_mat: torch.Tensor) -> torch.Tensor:
    _check_tensor(adj_mat)
    _check_square(adj_mat)

    if adj_mat.is_sparse:
        return norm_adj_mat_one_node_type_sparse(adj_mat)
    else:
        return norm_adj_mat_one_node_type_dense(adj_mat)


def norm_adj_mat_two_node_types_sparse(adj_mat: torch.Tensor) -> torch.Tensor:
    _check_tensor(adj_mat)
    _check_sparse(adj_mat)
    _check_2d(adj_mat)

    adj_mat = adj_mat.coalesce()
    indices = adj_mat.indices()
    values = adj_mat.values()
    degrees_row = torch.zeros(adj_mat.shape[0], device=adj_mat.device)
    degrees_row = degrees_row.index_add(0, indices[0], values.to(degrees_row.dtype))
    degrees_col = torch.zeros(adj_mat.shape[1], device=adj_mat.device)
    degrees_col = degrees_col.index_add(0, indices[1], values.to(degrees_col.dtype))
    values = values.to(degrees_row.dtype) / torch.sqrt(degrees_row[indices[0]] * degrees_col[indices[1]])
    adj_mat = _sparse_coo_tensor(indices, values, adj_mat.shape)

    return adj_mat


def norm_adj_mat_two_node_types_dense(adj_mat: torch.Tensor) -> torch.Tensor:
    _check_tensor(adj_mat)
    _check_dense(adj_mat)
    _check_2d(adj_mat)

    degrees_row = adj_mat.sum(1).view(-1, 1).to(torch.float32)
    degrees_col = adj_mat.sum(0).view(1, -1).to(torch.float32)
    degrees_row = torch.sqrt(degrees_row)
    degrees_col = torch.sqrt(degrees_col)
    adj_mat = adj_mat.to(degrees_row.dtype) / degrees_row
    adj_mat = adj_mat / degrees_col

    return adj_mat


def norm_adj_mat_two_node_types(adj_mat: torch.Tensor) -> torch.Tensor:
    _check_tensor(adj_mat)
    _check_2d(adj_mat)

    if adj_mat.is_sparse:
        return norm_adj_mat_two_node_types_sparse(adj_mat)
    else:
        return norm_adj_mat_two_node_types_dense(adj_mat)
