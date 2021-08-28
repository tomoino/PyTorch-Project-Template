# -*- coding: utf-8 -*-
"""Metrics

These functions are for metrics.

"""

from trainers.metrics.default_metrics import DefaultMetrics
from trainers.metrics.classification_metrics import ClassificationMetrics


SUPPORTED_METRICS = {
    "default": DefaultMetrics,
    "classification": ClassificationMetrics,
}


def get_metrics(cfg: object) -> object:
    """Get metrics

    This is function to get criterion.

    Args:
        cfg: Config.

    Returns:
        Metric object.

    Raises:
        NotImplementedError: If the metric you want to use is not suppoeted.

    """
    
    metrics_name = cfg.train.metrics.name

    if not metrics_name:
        return None

    if metrics_name not in SUPPORTED_METRICS:
        raise NotImplementedError('The metrics is not supported.')

    return SUPPORTED_METRICS[metrics_name](cfg)
