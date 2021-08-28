# -*- coding: utf-8 -*-
"""CIFAR10 dataset"""

import torch
from torchvision import datasets, transforms
from omegaconf import DictConfig

from data.dataset.base_dataset import BaseDataset
from data.dataset.helper import *


class CIFAR10(BaseDataset):
    """CIFAR10 dataset"""


    def __init__(self, cfg: DictConfig, mode: str) -> None:
        """Initialization
    
        Get CIFAR10 dataset.

        Args:
            cfg: Config.
            mode: Mode. 
                trainval: For trainning and validation.
                test: For test.

        """

        super().__init__(cfg, mode)

        _transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.classes = cfg.data.dataset.classes
        self.num_class = cfg.data.dataset.num_class

        if mode == "trainval":
            dataset = datasets.CIFAR10(
                root=cfg.data.dataset.rootdir,
                train=True,
                download = True,
                transform=_transform
            )
            
            num_shot = cfg.data.dataset.num_train_samples / self.num_class
            self.train, self.val = classification_train_val_split(dataset=dataset, num_shot=num_shot)

        elif mode == "test":
            self.test = datasets.CIFAR10(
                root = cfg.data.dataset.rootdir,
                train=False,
                download = True,
                transform=_transform
            )

            