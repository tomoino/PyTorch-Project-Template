# -*- coding: utf-8 -*-
"""SimpleCNN"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base_model import BaseModel


class Net(nn.Module):
    """Network for SimpleCNN"""


    def __init__(self, in_channel, out_channel) -> None:
        """Initialization

        Args:
            in_channel: Channel of input.
            out_channel: Channel of output.

        """

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_channel)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(BaseModel):
    """SimpleCNN"""


    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Build model.

        Args:
            cfg: Config.

        """

        super().__init__(cfg)
        self.num_class = self.cfg.data.dataset.num_class
        pretrained = self.cfg.model.pretrained

        self.network = Net(in_channel=self.cfg.data.dataset.in_channel, out_channel=self.num_class)

        self.build()