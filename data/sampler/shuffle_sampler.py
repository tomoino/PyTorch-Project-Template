# -*- coding: utf-8 -*-
"""Shuffle sampler"""

import logging

from omegaconf import DictConfig
from torch.utils.data.sampler import BatchSampler, RandomSampler

from data.sampler.base_sampler import BaseSampler

log = logging.getLogger(__name__)


class ShuffleSampler(BaseSampler):
    """Shuffle sampler

    This module is inherited by all sampler modules.

    Attributes:
        cfg: Config.
        network: Network object.
        device: Device. torch.device('cuda') or torch.device('cpu').
        optimizer: Optimizer object.
        criterion: Criterion object.

    """

    def __init__(self, cfg: DictConfig, mode: str, dataset: object) -> None:
        """Initialization

        Args:
            cfg: Config.
            mode: Mode. 
                trainval: For trainning and validation.
                test: For test.

        """

        super().__init__()
        if mode == "trainval":
            self.train = BatchSampler(RandomSampler(dataset.train), batch_size=cfg.train.batch_size, drop_last=True)
            self.val = BatchSampler(RandomSampler(dataset.val), batch_size=cfg.train.batch_size, drop_last=True)

        elif mode == "test":
            self.test = BatchSampler(RandomSampler(dataset.test), batch_size=cfg.train.batch_size, drop_last=True)
