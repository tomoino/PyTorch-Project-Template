# -*- coding: utf-8 -*-
"""Dataset"""
from omegaconf import DictConfig

from configs.supported_info import SUPPORTED_DATASET
from data.dataset.cifar10 import CIFAR10

def get_dataset(cfg: DictConfig, mode: str):
        
    dataset_name = cfg.data.dataset.name
    if dataset_name not in SUPPORTED_DATASET:
        raise NotImplementedError('The dataset is not supported.')
                    
    elif dataset_name == "cifar10":
        return CIFAR10(cfg, mode)
        