# -*- coding: utf-8 -*-
"""Omniglot dataset

"""

from torchvision import datasets, transforms

class Omniglot(datasets.Omniglot):
    def __init__(self, cfg: dict, mode: str):
        """Initialization
    
        Get Omniglot dataset.

        Args:
            cfg: Config.
            mode: Mode of data loader. 
                trainval: For trainning and validation.
                test: For test.

        """

        if mode == "trainval":
            super().__init__(
                background = True,
                root = cfg["data"]["dataset_root"],
                download = True,
                transform=transforms.ToTensor()
                )
        elif mode == "test":
            super().__init__(
                background = False,
                root = cfg["data"]["dataset_root"],
                download = True,
                transform=transforms.ToTensor()
                )