# -*- coding: utf-8 -*-
"""Metrics

These functions are for metrics.

"""

from configs.supported_info import SUPPORTED_METRIC
from metrics.classification_metric import ClassificationMetric


def get_metric(cfg: dict) -> object:
    """Get metric

    This is function to get criterion.

    Args:
        cfg: Config.

    Returns:
        Metric object.

    Raises:
        NotImplementedError: If the metric you want to use is not suppoeted.

    """
    
    metric_name = cfg["train"]["metric"]["name"]

    if metric_name not in SUPPORTED_METRIC:
        NotImplementedError('The metric is not supported.')

    if metric_name == "classification":
        return ClassificationMetric(cfg)