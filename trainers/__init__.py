# -*- coding: utf-8 -*-
"""Executor

These functions are for execution.

"""

from trainers.default_trainer import DefaultTrainer


SUPPORTED_TRAINER = {
    "default": DefaultTrainer,
}


def get_trainer(cfg: object) -> object:
    """Get trainer

    Args:
        cfg: Config of the project.

    Returns:
        Trainer object.

    Raises:
        NotImplementedError: If the model you want to use is not suppoeted.

    """

    trainer_name = cfg.train.trainer.name

    if trainer_name not in SUPPORTED_TRAINER:
        raise NotImplementedError('The trainer is not supported.')

    return SUPPORTED_TRAINER[trainer_name](cfg)
        