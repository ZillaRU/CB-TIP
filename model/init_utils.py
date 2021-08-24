#
# Copyright (C) Stanislaw Adaszewski, 2020
# License: GPLv3
#


import torch
import numpy as np


def init_glorot(in_channels, out_channels, dtype=torch.float32):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (in_channels + out_channels))
    initial = -init_range + 2 * init_range * \
              torch.rand(( in_channels, out_channels ), dtype=dtype)
    initial = initial.requires_grad_(True)
    return initial
