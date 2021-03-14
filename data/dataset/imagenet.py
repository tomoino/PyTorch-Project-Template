# -*- coding: utf-8 -*-
"""ImageNet dataset

"""

from torchvision import datasets, transforms

class ImageNet(datasets.ImageNet):
    def __init__(self, cfg: dict, mode: str):
        """Initialization
    
        Get ImageNet dataset.

        Args:
            cfg: Config.
            mode: Mode of data loader. 
                trainval: For trainning and validation.
                test: For test.

        """

        if mode == "trainval":
            super().__init__(
                root=cfg["data"]["dataset_root"],
                split="train",
                transform=transforms.ToTensor()
                )
        elif mode == "test":
            super().__init__(
                root = cfg["data"]["dataset_root"],
                split="val",
                transform=transforms.ToTensor()
                )