# -*- coding: utf-8 -*-
"""Helper functions on data

These functions are for handling data.

"""

from torch.utils.data import Subset


def class_filter(dataset: object, classes: list) -> object:
    """Fileters dataset for classification

    Filters to create a dataset that contains only the classes you need.

    Args:
        dataset: Dataset.
        classes: List of classes. 

    Returns:
        Dataset object.

    """

    dataset_len = len(dataset)
    indices = []

    for idx in range(dataset_len):
        if dataset[idx][1] in classes:
            indices.append(idx)

    filtered_dataset = Subset(dataset, indices)

    return filtered_dataset


def classification_train_val_split(dataset: object, num_shot: int) -> tuple:
    """Train-val splitter for classification

    Split dataset to train dataset and validation dataset for classification.

    Args:
        dataset: Dataset.
        num_shot: Number of samples of train data for each class.

    Returns:
        Tuple of dataset objects.

    """

    dataset_len = len(dataset)
    num_samples_per_class = {}
    train_indices = []
    val_indices = []

    for idx in range(dataset_len):
        class_label = dataset[idx][1]
        if class_label not in num_samples_per_class:
            num_samples_per_class[class_label] = 1
            train_indices.append(idx)
            
        elif num_samples_per_class[class_label] < num_shot:
            train_indices.append(idx)
            num_samples_per_class[class_label] += 1

        else:
            val_indices.append(idx)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset