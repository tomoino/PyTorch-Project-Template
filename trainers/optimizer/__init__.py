# -*- coding: utf-8 -*-
"""Optimizer

These functions are for optimizers.

"""

import torch.optim as optim
import torch.nn as nn


SUPPORTED_OPTIMIZER = {
    "adam": None,
}


def get_optimizer(cfg: object, network: object) -> object:
    """Get optimizer function

    This is function to get optimizer.

    Args:
        cfg: Config of optimizer.
        network: Network of model.

    Returns:
        Optimizer object.

    Raises:
        NotImplementedError: If the optimizer you want to use is not suppoeted.

    """
    
    optimizer_name = cfg.train.optimizer.name

    if not optimizer_name:
        return None

    if optimizer_name not in SUPPORTED_OPTIMIZER:
        raise NotImplementedError('The optimizer is not supported.')

    if optimizer_name == "adam":
        return optim.Adam(network.parameters(),
                          lr=cfg.train.optimizer.lr,
                          weight_decay=cfg.train.optimizer.decay)

    else:
        return SUPPORTED_OPTIMIZER[optimizer_name](cfg, network)
