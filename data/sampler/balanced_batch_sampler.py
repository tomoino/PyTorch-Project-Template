# -*- coding: utf-8 -*-
"""Balanced Batch Sampler

"""

from torch.utils.data.sampler import BatchSampler, RandomSampler


class BalancedBatchSampler(BatchSampler):
    def __init__(self, cfg: dict, dataset: object):
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
            batch_size = cfg["train"]["batch_size"],
            drop_last = True
            )
        self.class_num = cfg["train"]["class_num"]
        self.class_mini_batch_size = self.batch_size // self.class_num

    def __iter__(self):
        batches = []
        checked_samples_num = {}
        for idx in self.sampler:
            class_id = self.dataset[idx][1]
            if class_id not in checked_samples_num:
                checked_samples_num[class_id] = 0
            target_batch_id = checked_samples_num[class_id] // self.class_mini_batch_size
            if len(batches) == target_batch_id:
                batches.append([])
            batches[target_batch_id].append(idx)
            checked_samples_num[class_id] += 1
            if len(batches[target_batch_id]) == self.batch_size:
                yield batches[target_batch_id]

    def __len__(self):
        return len(self.sampler) // self.batch_size