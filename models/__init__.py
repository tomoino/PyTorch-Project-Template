# -*- coding: utf-8 -*-
"""Models module

This is the module for models.

"""

from models.networks.resnet18 import ResNet18
from models.networks.simple_cnn import SimpleCNN


SUPPORTED_MODEL = {
    "resnet18": ResNet18,
    "simple_cnn": SimpleCNN,
}


def get_model(cfg: object) -> object:
    """Get model function

    This is function to get model.

    Args:
        cfg: Config of the project.

    Returns:
        Model object.

    Raises:
        NotImplementedError: If the model you want to use is not suppoeted.

    """

    model_name = cfg.model.name

    if model_name not in SUPPORTED_MODEL:
        raise NotImplementedError('The model is not supported.')

    return SUPPORTED_MODEL[model_name](cfg)
