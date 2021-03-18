# -*- coding: utf-8 -*-
"""ResNet18"""

import torch.nn as nn
import torchvision.models as models

from models.base_model import BaseModel


class ResNet18(BaseModel):
    """ResNet18"""

    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Build model.

        Args:
            cfg: Config.

        """

        super().__init__(cfg)
        self.num_class = self.cfg.data.dataset.num_class
        pretrained = self.cfg.model.pretrained

        self.network = models.resnet18(pretrained=pretrained)
        self.network.fc = nn.Linear(in_features = 512, out_features = self.num_class, bias = True)

        self.build()