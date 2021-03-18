# -*- coding: utf-8 -*-
"""Data module

This is the module for handling data.

"""

import logging

from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from configs.supported_info import SUPPORTED_DATASET, SUPPORTED_SAMPLER
import data.helper
from data.dataloader import DataLoader
from data.dataset.omniglot import Omniglot
from data.dataset.cifar10 import CIFAR10
from data.sampler.balanced_batch_sampler import BalancedBatchSampler


log = logging.getLogger(__name__)


def get_dataset(cfg: object, mode: str) -> tuple:
    """Get dataset function

    This is function to get dataset.

    Args:
        cfg: Config.
        mode: Mode of dataset. 
            trainval: For trainning and validation.
            test: For test.

    Returns:
        Tuple of dataset objects.

    Raises:
        NotImplementedError: If the dataset you want to use is not suppoeted.

    """

    dataset_name = cfg.data.dataset.name

    if dataset_name not in SUPPORTED_DATASET:
        raise NotImplementedError('The dataset is not supported.')

    if dataset_name == "omniglot":
        dataset = Omniglot(cfg, mode)
        classes = list(range(cfg.data.dataset.num_class))
        filtered_dataset = helper.class_filter(dataset=dataset, classes=classes)

        if mode == "trainval":
            num_shot = cfg.data.dataset.num_train_samples / cfg.data.dataset.num_class
            return helper.classification_train_val_split(dataset=filtered_dataset, num_shot=num_shot)
        elif mode == "test":
            return filtered_dataset
            
    elif dataset_name == "cifar10":
        dataset = CIFAR10(cfg, mode)
        classes = list(range(cfg.data.dataset.num_class))
        filtered_dataset = helper.class_filter(dataset=dataset, classes=classes)

        if mode == "trainval":
            num_shot = cfg.data.dataset.num_train_samples / cfg.data.dataset.num_class
            return helper.classification_train_val_split(dataset=filtered_dataset, num_shot=num_shot)
        elif mode == "test":
            return filtered_dataset


def get_sampler(cfg: object, mode: str, dataset: object) -> object:
    """Get sampler function

    This is function to get samplers.

    Args:
        cfg: Config.
        mode: Mode. 
            train: For trainning.
            val: For validation.
            test: For test.
        dataset: Dataset.

    Returns:
        Sampler.

    Raises:
        NotImplementedError: If the sampler you want to use is not suppoeted.

    """

    sampler_name = cfg.data.sampler.name

    if sampler_name not in SUPPORTED_SAMPLER:
        raise NotImplementedError('The sampler is not supported.')

    if sampler_name == "shuffle_sampler":
        return BatchSampler(RandomSampler(dataset), batch_size=cfg.train.batch_size, drop_last=True)
        
    elif sampler_name == "balanced_batch_sampler":
        if mode == "train":
            return BalancedBatchSampler(cfg, dataset=dataset)

        elif mode == "val" or mode == "test":
            return BatchSampler(SequentialSampler(dataset), batch_size=cfg.train.batch_size, drop_last=False)


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

    if mode == "trainval":
        train_dataset, val_dataset = get_dataset(cfg, mode="trainval")
        train_sampler = get_sampler(cfg, mode="train", dataset=train_dataset)
        val_sampler = get_sampler(cfg, mode="val", dataset=val_dataset)
        train_dataloader = DataLoader(cfg, dataset=train_dataset, sampler=train_sampler)
        val_dataloader = DataLoader(cfg, dataset=val_dataset, sampler=val_sampler)

        log.info(f"Successfully loaded {cfg.data.dataset.name} dataset.")

        return train_dataloader, val_dataloader

    elif mode == "test":
        test_dataset = get_dataset(cfg, mode="test")
        test_sampler = get_sampler(cfg, mode="test", dataset=test_dataset)
        test_dataloader = DataLoader(cfg, dataset=test_dataset, sampler=test_sampler)

        log.info(f"Successfully loaded {cfg.data.dataset.name} dataset.")
        
        return test_dataloader