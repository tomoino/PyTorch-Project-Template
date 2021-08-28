# -*- coding: utf-8 -*-
"""Abstract base dataset"""

import logging

from abc import ABC
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base dataset

    This module is inherited by all dataset modules.

    Attributes:
        cfg: Config.
        network: Network object.
        device: Device. torch.device('cuda') or torch.device('cpu').
        optimizer: Optimizer object.
        criterion: Criterion object.

    """

    def __init__(self, cfg: DictConfig, mode: str) -> None:
        """Initialization

        Args:
            cfg: Config.
            mode: Mode. 
                trainval: For trainning and validation.
                test: For test.

        """

        self.cfg = cfg
        self.mode = mode
        self.train = None
        self.val = None
        self.test = None