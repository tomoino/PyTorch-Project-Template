# -*- coding: utf-8 -*-
"""CIFAR10 dataset"""

import torch
from torchvision import datasets, transforms


class CIFAR10(datasets.CIFAR10):
    """CIFAR10 dataset"""


    def __init__(self, cfg: object, mode: str) -> None:
        """Initialization
    
        Get CIFAR10 dataset.

        Args:
            cfg: Config.
            mode: Mode. 
                trainval: For trainning and validation.
                test: For test.

        """

        _transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if mode == "trainval":
            super().__init__(
                root=cfg.data.dataset.rootdir,
                train=True,
                download = True,
                transform=_transform
                )
        elif mode == "test":
            super().__init__(
                root = cfg.data.dataset.rootdir,
                train=False,
                download = True,
                transform=_transform
                )