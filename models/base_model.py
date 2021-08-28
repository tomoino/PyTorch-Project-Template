# -*- coding: utf-8 -*-
"""Abstract base model"""

import logging
from pathlib import Path

import torch
from abc import ABC


log = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base model

    This model is inherited by all models.

    Attributes:
        cfg: Config.
        network: Network object.
        device: Device. torch.device('cuda') or torch.device('cpu').
        optimizer: Optimizer object.
        criterion: Criterion object.

    """

    def __init__(self, cfg: object) -> None:
        """Initialization

        Args:
            cfg: Config.

        """

        log.info(f'Building {cfg.model.name} model...')

        self.cfg = cfg
        self.network = None
        self.device = None


    def build(self) -> None:
        """Build model

        Note:
            Use this function after setting model.

        """

        self.load_ckpt()
        self.setup_device()

        log.info(f"Successfully built {self.cfg.model.name} model.")


    def load_ckpt(self) -> None:
        """Loads checkpoint

        Raises:
            ValueError: If the ckpt isn't found.        

        """

        initial_ckpt = self.cfg.model.initial_ckpt

        if not initial_ckpt:
            return

        ckpt_path = Path(initial_ckpt)
        
        if not ckpt_path.exists():
            raise ValueError(' The checkpoint is not found.')

        ckpt = torch.load(ckpt_path)
        self.network.load_state_dict(ckpt['model_state_dict'])


    def setup_device(self) -> None:
        """Setup device"""

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.network = self.network.to(self.device)
