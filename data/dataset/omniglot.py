# -*- coding: utf-8 -*-
"""Omniglot dataset"""

from torchvision import datasets, transforms

class Omniglot(datasets.Omniglot):
    """Omniglot dataset"""


    def __init__(self, cfg: object, mode: str) -> None:
        """Initialization
    
        Get Omniglot dataset.

        Args:
            cfg: Config.
            mode: Mode. 
                trainval: For trainning and validation.
                test: For test.

        """

        if mode == "trainval":
            super().__init__(
                background = True,
                root = cfg.data.dataset.rootdir,
                download = True,
                transform=transforms.ToTensor()
                )
        elif mode == "test":
            super().__init__(
                background = False,
                root = cfg.data.dataset.rootdir,
                download = True,
                transform=transforms.ToTensor()
                )