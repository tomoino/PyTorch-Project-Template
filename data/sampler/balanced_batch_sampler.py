# -*- coding: utf-8 -*-
"""Balanced Batch Sampler"""

from omegaconf import DictConfig

from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
from data.sampler.base_sampler import BaseSampler


class _BalancedBatchSampler(BatchSampler):
    """Balanced Batch Sampler"""


    def __init__(self, cfg: object, dataset: object) -> None:
        """Initialization
    
        Balanced batch sampler.
        Create a batch so that the number of samples for each class is equal.
        This sampler will drop the last batch if its size would be less than batch_size.

        Args:
            cfg: Config.
            dataset: object

        """

        self.dataset = dataset
        super().__init__(
            sampler = RandomSampler(self.dataset),
            batch_size = cfg.train.batch_size,
            drop_last = True
            )
        self.num_class = cfg.data.dataset.num_class
        self.class_mini_batch_size = self.batch_size // self.num_class

    def __iter__(self):
        batches = []
        num_checked_samples = {}
        for idx in self.sampler:
            class_id = self.dataset[idx][1]
            if class_id not in num_checked_samples:
                num_checked_samples[class_id] = 0
            target_batch_id = num_checked_samples[class_id] // self.class_mini_batch_size
            if len(batches) == target_batch_id:
                batches.append([])
            batches[target_batch_id].append(idx)
            num_checked_samples[class_id] += 1
            if len(batches[target_batch_id]) == self.batch_size:
                yield batches[target_batch_id]

    def __len__(self) -> int:
        return len(self.sampler) // self.batch_size


class BalancedBatchSampler(BaseSampler):
    """Balanced Batch Sampler"""

    def __init__(self, cfg: DictConfig, mode: str, dataset: object) -> None:
        """Initialization

        Args:
            cfg: Config.
            mode: Mode. 
                trainval: For trainning and validation.
                test: For test.

        """

        super().__init__(cfg, mode)
        if mode == "trainval":
            self.train = _BalancedBatchSampler(cfg, dataset=dataset.train)
            self.val = BatchSampler(SequentialSampler(dataset.val), batch_size=cfg.train.batch_size, drop_last=False)

        elif mode == "test":
            self.test = BatchSampler(SequentialSampler(dataset.test), batch_size=cfg.train.batch_size, drop_last=False)