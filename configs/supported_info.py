"""Supported infomation

Information about supported datasets, samplers, models, optimizers, criterions and metrics.

"""


SUPPORTED_DATASET = [
    "cifar10",
]

SUPPORTED_SAMPLER = [
    "shuffle_sampler",
    "balanced_batch_sampler",
]

SUPPORTED_MODEL = [
    "resnet18",
    "simple_cnn",
]

SUPPORTED_OPTIMIZER = [
    "adam",
]

SUPPORTED_CRITERION = [
    "cross_entropy",
]

SUPPORTED_METRICS = [
    "classification",
]

SUPPORTED_TRAINER = [
    "default",
    "classification",
]