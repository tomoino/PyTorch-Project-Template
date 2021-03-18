# -*- coding: utf-8 -*-
"""Models helper

These are helper functions for models.

"""

import torch.optim as optim
import torch.nn as nn

from configs.supported_info import SUPPORTED_OPTIMIZER, SUPPORTED_CRITERION


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
    
    optimizer_name = cfg.name

    if not optimizer_name:
        return None

    if optimizer_name not in SUPPORTED_OPTIMIZER:
        raise NotImplementedError('The optimizer is not supported.')

    if optimizer_name == "adam":
        return optim.Adam(network.parameters(),
                          lr=cfg.lr,
                          weight_decay=cfg.decay)


def get_criterion(cfg: object) -> object:
    """Get criterion function

    This is function to get criterion.

    Args:
        cfg: Config of criterion.

    Returns:
        Criterion object.

    Raises:
        NotImplementedError: If the criterion you want to use is not suppoeted.

    """
    
    criterion_name = cfg.name

    if not criterion_name:
        return None

    if criterion_name not in SUPPORTED_CRITERION:
        raise NotImplementedError('The loss function is not supported.')

    if criterion_name == "cross_entropy":
        return nn.CrossEntropyLoss()