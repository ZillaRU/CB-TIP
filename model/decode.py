#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch
from init_utils import init_glorot
from dropout_utils import dropout
from typing import Tuple, \
    List


def dedicom_decoder(input_dim: int, num_relation_types: int) -> \
        Tuple[torch.Tensor, List[torch.Tensor]]:

    global_interaction = init_glorot(input_dim, input_dim)
    local_variation = [
        torch.diag(torch.flatten(init_glorot(input_dim, 1))) \
        for _ in range(num_relation_types)
    ]
    return (global_interaction, local_variation)


def dist_mult_decoder(input_dim: int, num_relation_types: int) -> \
        Tuple[torch.Tensor, List[torch.Tensor]]:

    global_interaction = torch.eye(input_dim, input_dim)
    local_variation = [
        torch.diag(torch.flatten(init_glorot(input_dim, 1))) \
        for _ in range(num_relation_types)
    ]
    return (global_interaction, local_variation)


def bilinear_decoder(input_dim: int, num_relation_types: int) -> \
        Tuple[torch.Tensor, List[torch.Tensor]]:

    global_interaction = torch.eye(input_dim, input_dim)
    local_variation = [
        init_glorot(input_dim, input_dim) \
        for _ in range(num_relation_types)
    ]
    return (global_interaction, local_variation)


def inner_product_decoder(input_dim: int, num_relation_types: int) -> \
        Tuple[torch.Tensor, List[torch.Tensor]]:

    global_interaction = torch.eye(input_dim, input_dim)
    local_variation = torch.eye(input_dim, input_dim)
    local_variation = [ local_variation ] * num_relation_types
    return (global_interaction, local_variation)
