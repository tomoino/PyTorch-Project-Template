# -*- coding: utf-8 -*-
"""Abstract base metrics"""

import logging
from abc import ABC

import torch
import mlflow


log = logging.getLogger(__name__)


class BaseMetrics(ABC):
    """Abstract base metrics

    This model is inherited by all metrics modules.

    Attributes:
        cfg: Config of project.

    """

    def __init__(self, cfg: object, init_best_score: float) -> None:
        """Initialization

        Args:
            cfg: Config of project.

        """

        self.cfg = cfg
        self.loss_list = []
        self.best_score = init_best_score


    def batch_update(self, outputs, targets, loss) -> None:
        """Update metrics for each batch

        Args:
            outputs: Outputs.
            targets: Target values.
            loss: Loss.

        """

        self.loss_list.append(loss)

        
    def epoch_update(self, epoch: int, mode: str) -> None:
        """Calculate metrics for each epoch
        
        Calculates accuracy, loss, precision, recall and f1score.
        
        Args:
            epoch: Current epoch.
            mode: Mode. 
                train: For trainning.
                eval: For validation or test.
            
        """

        pass


    def reset_states(self) -> None:
        """Reset states"""

        self.loss_list = []


    def judge_update_ckpt(self) -> bool:
        """Judge whether ckpt should be updated or not"""

        pass
