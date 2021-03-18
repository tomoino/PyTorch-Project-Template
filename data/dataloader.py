# -*- coding: utf-8 -*-
"""DataLoader class"""

import torch


class DataLoader(torch.utils.data.DataLoader):
    """Dataloader
    
    This module is for data loader.

    """

    def __init__(self, cfg: object, dataset: object, sampler: object) -> None:
        """Initialization
    
        Get data loader.

        Args:
            cfg: Config.
            dataset: Dataset. 
            sampler: Batch sampler.

        """

        super().__init__(
            dataset=dataset,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            batch_sampler=sampler,
            )