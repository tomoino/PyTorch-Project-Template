# -*- coding: utf-8 -*-
"""Criterion

These functions are for criterion.

"""

import torch.optim as optim
import torch.nn as nn


SUPPORTED_CRITERION = {
    "cross_entropy": nn.CrossEntropyLoss,
}


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
    
    criterion_name = cfg.train.criterion.name

    if not criterion_name:
        return None

    if criterion_name not in SUPPORTED_CRITERION:
        raise NotImplementedError('The loss function is not supported.')

    return SUPPORTED_CRITERION[criterion_name]()
    