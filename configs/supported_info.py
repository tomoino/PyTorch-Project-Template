"""Supported datasets, samplers and models

Information about supported datasets, samplers and models.

Note:
    You can add datasets, samplers or models.

    When you want to add a dataset, you have to 
        modify this file: SUPPORTED_DATASET
        modify data/__init__.py: get_dataset function
        add modeule to data/dataset/

    When you want to add a sampler, you have to 
        modify this file: SUPPORTED_SAMPLER
        modify data/__init__.py: get_sampler function
        add modeule to data/sampler/

"""


SUPPORTED_DATASET = [
    "omniglot",
]

SUPPORTED_SAMPLER = [
    "balanced_batch_sampler",
]

SUPPORTED_MODEL = [
    "resnet18",
]

SUPPORTED_OPTIMIZER = [
    "adam",
]

SUPPORTED_CRITERION = [
    "cross_entropy",
]

SUPPORTED_METRIC = [
    "classification",
]