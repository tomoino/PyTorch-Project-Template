# -*- coding: utf-8 -*-
"""Abstract base model

This model is inherited by all models.

"""

from pathlib import Path

import torch
from abc import ABC

from models.helper import get_optimizer, get_criterion
from metrics import get_metric


class BaseModel(ABC):

    def __init__(self, cfg: dict):
        self.config = cfg
        self.model = None
        self.device = None
        self.optimizer = None
        self.criterion = None


    def build(self) -> None:
        """Build model

        Note:
            Use this function after setting model.

        """

        self.load_ckpt()
        self.setup_device()
        self.set_optimizer()
        self.set_criterion()
        self.set_metric()

    def load_ckpt(self) -> None:
        """Loads checkpoint

        Raises:
            ValueError: If the ckpt isn't found.        

        """

        if not self.config["model"]["initial_ckpt"]:
            return

        ckpt_path = Path(self.config["model"]["initial_ckpt"])
        
        if not ckpt_path.exists():
            raise ValueError(' The checkpoint is not found.')

        ckpt = torch.load(resume)
        self.model.load_state_dict(ckpt['model_state_dict'])

    def setup_device(self) -> None:
        """Setup device """

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = self.model.to(self.device)

    def set_optimizer(self) -> None:
        """Set optimizer """
        self.optimizer = get_optimizer(self.config, self.model)

    def set_criterion(self) -> None:
        """Set criterion """
        self.criterion = get_criterion(self.config)

    def set_metric(self) -> None:
        """Set metric """
        self.metric = get_metric(self.config)