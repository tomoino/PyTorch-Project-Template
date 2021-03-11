# -*- coding: utf-8 -*-
"""Models helper

These are helper functions for models.

"""

import torch.optim as optim
import torch.nn as nn

from configs.supported_info import SUPPORTED_OPTIMIZER, SUPPORTED_CRITERION


def get_optimizer(cfg: dict, model: object) -> object:
    """Get optimizer function

    This is function to get optimizer.

    Args:
        cfg: Config.
        model: Model.

    Returns:
        Optimizer object.

    Raises:
        NotImplementedError: If the optimizer you want to use is not suppoeted.

    """
    
    optimizer_name = cfg["train"]["optimizer"]["name"]

    if optimizer_name not in SUPPORTED_OPTIMIZER:
        NotImplementedError('The optimizer is not supported.')

    if optimizer_name == "adam":
        return optim.Adam(model.parameters(),
                          lr=cfg["train"]["optimizer"]["lr"],
                          weight_decay=cfg["train"]["optimizer"]["decay"])

def get_criterion(cfg: dict) -> object:
    """Get criterion function

    This is function to get criterion.

    Args:
        cfg: Config.

    Returns:
        Criterion object.

    Raises:
        NotImplementedError: If the criterion you want to use is not suppoeted.

    """
    
    criterion_name = cfg["train"]["criterion"]["name"]

    if criterion_name not in SUPPORTED_CRITERION:
        NotImplementedError('The loss function is not supported.')

    if criterion_name == "cross_entropy":
        return nn.CrossEntropyLoss()