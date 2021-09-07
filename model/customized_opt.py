from argparse import Namespace

import torch.nn as nn
from torch.optim import Optimizer, Adam


def build_optimizer(encoder: nn.Module,
                    decoder1: nn.Module,
                    decoder2: nn.Module,
                    MIM_model: nn.Module,
                    args: Namespace) -> Optimizer:
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """
    params = [
        # {'params': encoder.parameters(), 'lr': args.enc_lr, 'weight_decay': 0},
        # {'params': decoder.parameters(), 'lr': args.dec_lr, 'weight_decay': 0},
        # {'params': MIM_model.parameters(), 'lr': args.mim_lr, 'weight_decay': 0}
        {'params': encoder.parameters(), 'lr': args.learning_rate, 'weight_decay': 0},
        {'params': decoder1.parameters(), 'lr': args.learning_rate, 'weight_decay': 0},
        {'params': decoder2.parameters(), 'lr': args.learning_rate, 'weight_decay': 0},
        {'params': MIM_model.parameters(), 'lr': args.learning_rate, 'weight_decay': 0}
    ]

    return Adam(params)

    # opt = torch.optim.Adam([
    #     {'params': model.parameters()}
    #     # {'params': loss_fuc.parameters(), 'lr': 10 * params.learning_rate},
    #     # {'params': awl_dd.parameters(), 'lr': 1 * params.learning_rate},
    #     # {'params': awl_dt.parameters(), 'lr': 10 * params.learning_rate},
    #     # {'params': awl_tt.parameters(), 'lr': 1 * params.learning_rate}
    # ], lr=params.learning_rate)
