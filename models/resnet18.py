# -*- coding: utf-8 -*-
"""ResNet18

"""

import torch.nn as nn
import torchvision.models as models

from models.base_model import BaseModel


class ResNet18(BaseModel):
    def __init__(self, cfg: dict):
        """Initialization
    
        Build model.

        Args:
            cfg: Config.

        """

        super().__init__(cfg)
        self.class_num = self.config["train"]["class_num"]
        pretrained = self.config["model"]["pretrained"]

        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(in_features = 512, out_features = self.class_num, bias = True)

        self.build()