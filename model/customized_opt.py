from argparse import Namespace

import torch.nn as nn
from torch.optim import Optimizer, Adam


def build_optimizer(encoder: nn.Module,
                    decoder1: nn.Module,
                    decoder2: nn.Module,
                    MIM_model: nn.Module,
                    args: Namespace) -> Optimizer:
    params = [
        {'params': encoder.parameters(), 'lr': args.learning_rate, 'weight_decay': 0},
        {'params': decoder1.parameters(), 'lr': args.learning_rate, 'weight_decay': 0},
        {'params': decoder2.parameters(), 'lr': args.learning_rate, 'weight_decay': 0},
        {'params': MIM_model.parameters(), 'lr': args.learning_rate, 'weight_decay': 0}
    ]

    return Adam(params)
