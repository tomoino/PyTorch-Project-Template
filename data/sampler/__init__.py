# -*- coding: utf-8 -*-
"""Sampler"""
from omegaconf import DictConfig
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from data.sampler.balanced_batch_sampler import BalancedBatchSampler
from data.sampler.shuffle_sampler import ShuffleSampler


SUPPORTED_SAMPLER = {
    "shuffle_sampler": ShuffleSampler,
    "balanced_batch_sampler": BalancedBatchSampler,
}


def get_sampler(cfg: DictConfig, mode: str, dataset):
        
    sampler_name = cfg.data.sampler.name

    if sampler_name not in SUPPORTED_SAMPLER:
        raise NotImplementedError('The sampler is not supported.')

    return SUPPORTED_SAMPLER[sampler_name](cfg, mode, dataset)
