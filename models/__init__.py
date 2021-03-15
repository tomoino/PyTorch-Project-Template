# -*- coding: utf-8 -*-
"""Models module

This is the module for models.

"""

from configs.supported_info import SUPPORTED_MODEL
from models.resnet18 import ResNet18


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
        NotImplementedError('The model is not supported.')

    if model_name == "resnet18":
        return ResNet18(cfg)