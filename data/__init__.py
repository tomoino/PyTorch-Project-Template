# -*- coding: utf-8 -*-
"""Data module

This is the module for handling data.

"""

import logging

from data.dataloader import DataLoader
from data.dataset import get_dataset
from data.sampler import get_sampler


log = logging.getLogger(__name__)


def get_dataloader(cfg: object, mode: str) -> tuple:
    """Get dataloader function

    This is function to get dataloaders.
    Get dataset, then make dataloaders.

    Args:
        cfg: Config.
        mode: Mode. 
            trainval: For trainning and validation.
            test: For test.

    Returns:
        Tuple of dataloaders.

    """

    log.info(f"Loading {cfg.data.dataset.name} dataset...")

    dataset = get_dataset(cfg, mode)
    sampler = get_sampler(cfg, mode, dataset)

    if mode == "trainval":
        train_dataloader = DataLoader(cfg, dataset=dataset.train, sampler=sampler.train)
        val_dataloader = DataLoader(cfg, dataset=dataset.val, sampler=sampler.val)
        dataloaders = (train_dataloader, val_dataloader)

    elif mode == "test":
        test_dataloader = DataLoader(cfg, dataset=dataset.test, sampler=sampler.test)
        dataloaders = (test_dataloader)

    log.info(f"Successfully loaded {cfg.data.dataset.name} dataset.")
        
    return dataloaders
