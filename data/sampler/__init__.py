# -*- coding: utf-8 -*-
"""Sampler"""
from omegaconf import DictConfig
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from configs.supported_info import SUPPORTED_SAMPLER
from data.sampler.balanced_batch_sampler import BalancedBatchSampler
from data.sampler.shuffle_sampler import ShuffleSampler


def get_sampler(cfg: DictConfig, mode: str, dataset):
        
    sampler_name = cfg.data.sampler.name

    if sampler_name not in SUPPORTED_SAMPLER:
        raise NotImplementedError('The sampler is not supported.')

    if sampler_name == "shuffle_sampler":
        return ShuffleSampler(cfg, mode, dataset)
            
    elif sampler_name == "balanced_batch_sampler":
        return BalancedBatchSampler(cfg, mode, dataset)
