from argparse import Namespace

import torch.nn as nn
from torch.optim import Optimizer, Adam


def build_optimizer(model: nn.Module, args: Namespace) -> Optimizer:
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """
    params = [{'params': model.parameters(), 'lr': args.learning_rate, 'weight_decay': 0}]

    return Adam(params)

    # opt = torch.optim.Adam([
    #     {'params': model.parameters()}
    #     # {'params': loss_fuc.parameters(), 'lr': 10 * params.learning_rate},
    #     # {'params': awl_dd.parameters(), 'lr': 1 * params.learning_rate},
    #     # {'params': awl_dt.parameters(), 'lr': 10 * params.learning_rate},
    #     # {'params': awl_tt.parameters(), 'lr': 1 * params.learning_rate}
    # ], lr=params.learning_rate)