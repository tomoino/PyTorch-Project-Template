# -*- coding: utf-8 -*-
"""Data module

This is the module for handling data.

"""

from torch.utils.data.sampler import BatchSampler, SequentialSampler

from configs.supported_info import *
import data.helper
from data.dataloader import DataLoader
from data.dataset.omniglot import Omniglot
from data.sampler.balanced_batch_sampler import BalancedBatchSampler

def get_dataset(cfg: dict, mode: str) -> tuple:
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

    dataset_name = cfg["data"]["dataset_name"]

    if dataset_name not in SUPPORTED_DATASET:
        NotImplementedError('The dataset is not supported.')

    if dataset_name == "omniglot":
        _dataset = Omniglot(cfg, mode)
        _classes = list(range(cfg["train"]["class_num"]))
        filtered_dataset = helper.class_filter(dataset=_dataset, classes=_classes)

        if mode == "trainval":
            return helper.classification_train_val_split(dataset=filtered_dataset, shot_num=cfg["train"]["shot_num"])
        elif mode == "test":
            return filtered_dataset


def get_sampler(cfg: dict, mode: str, dataset: object) -> object:
    """Get sampler function

    This is function to get samplers.

    Args:
        cfg: Config.
        mode: Mode of sampler. 
            train: For trainning.
            val: For validation.
            test: For test.
        dataset: Dataset.

    Returns:
        Sampler.

    Raises:
        NotImplementedError: If the sampler you want to use is not suppoeted.

    """

    sampler_name = cfg["train"]["sampler_name"]

    if sampler_name not in SUPPORTED_SAMPLER:
        NotImplementedError('The sampler is not supported.')

    if sampler_name == "balanced_batch_sampler":
        if mode == "train":
            return BalancedBatchSampler(cfg, dataset=dataset)

        elif mode == "val" or mode == "test":
            return BatchSampler(SequentialSampler(dataset), batch_size=cfg["train"]["batch_size"], drop_last=False)


def get_dataloader(cfg: dict, mode: str) -> tuple:
    """Get dataloader function

    This is function to get dataloaders.
    Get dataset, then make dataloaders.

    Args:
        cfg: Config.
        mode: Mode of dataloader. 
            trainval: For trainning and validation.
            test: For test.

    Returns:
        Tuple of dataloaders.

    """

    if mode == "trainval":
        train_dataset, val_dataset = get_dataset(cfg, mode="trainval")
        train_sampler = get_sampler(cfg, mode="train", dataset=train_dataset)
        val_sampler = get_sampler(cfg, mode="val", dataset=val_dataset)
        train_dataloader = DataLoader(cfg, dataset=train_dataset, sampler=train_sampler)
        val_dataloader = DataLoader(cfg, dataset=val_dataset, sampler=val_sampler)

        return train_dataloader, val_dataloader

    elif mode == "test":
        test_dataset = get_dataset(cfg, mode="test")
        test_sampler = get_sampler(cfg, mode="test", dataset=test_dataset)
        test_dataloader = DataLoader(cfg, dataset=test_dataset, sampler=test_sampler)

        return test_dataloader