# -*- coding: utf-8 -*-
"""Default metrics"""

import logging
import math
from statistics import mean

import torch
import mlflow


log = logging.getLogger(__name__)


class DefaultMetrics:
    """Default metrics

    This module is simple metrics using loss.

    Attributes:
        loss_list: List of losses during 1 epoch.
        model_score: Indicator of the excellence of model. The higher the value, the better.

    Note:
        At the beginning of every epoch, loss_list must be empty.

    """


    def __init__(self, cfg: object) -> None:
        """Initialization

        Args:
            cfg: Config.

        """

        self.loss_list = []
        self.best_score = math.inf


    def batch_update(self, outputs, targets, loss) -> None:
        """Update loss and cmx

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
        
        loss = mean(self.loss_list)

        if mode == "train":
            metrics = {
                "loss": loss,
            }
            log.info(f"\tloss: {loss}")

        elif mode == "eval":
            metrics = {
                "val_loss": loss,
            }
            log.info(f"\tval_loss: {loss}")

        if mlflow.active_run():
            mlflow.log_metrics(metrics, step = epoch)
        self.model_score = loss
        self.reset_states()
        

    def reset_states(self) -> None:
        """Reset states

        Resets loss_list and cmx

        """

        self.loss_list = []


    def judge_update_ckpt(self) -> bool:
        """Judge whether ckpt should be updated or not"""
        
        if self.model_score < self.best_score:
            self.best_score = self.model_score
            
            return True
            
        else:
            return False