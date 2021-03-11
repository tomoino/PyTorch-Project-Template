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


def classification_train_val_split(dataset: object, shot_num: int) -> tuple:
    """Train-val splitter for classification

    Split dataset to train dataset and validation dataset for classification.

    Args:
        dataset: Dataset.
        shot_num: Number of samples of train data for each class.

    Returns:
        Tuple of dataset objects.

    """

    dataset_len = len(dataset)
    train_num_dict = {}
    train_indices = []
    val_indices = []

    for idx in range(dataset_len):
        class_label = dataset[idx][1]
        if class_label not in train_num_dict:
            train_num_dict[class_label] = 1
            train_indices.append(idx)
            
        elif train_num_dict[class_label] < shot_num:
            train_indices.append(idx)
            train_num_dict[class_label] += 1

        else:
            val_indices.append(idx)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset
