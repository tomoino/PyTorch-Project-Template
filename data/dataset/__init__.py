# -*- coding: utf-8 -*-
"""Dataset"""
from omegaconf import DictConfig

from data.dataset.cifar10 import CIFAR10


SUPPORTED_DATASET = {
    "cifar10": CIFAR10,
}


def get_dataset(cfg: DictConfig, mode: str):
        
    dataset_name = cfg.data.dataset.name
    if dataset_name not in SUPPORTED_DATASET:
        raise NotImplementedError('The dataset is not supported.')
                    
    return SUPPORTED_DATASET[dataset_name](cfg, mode)
        