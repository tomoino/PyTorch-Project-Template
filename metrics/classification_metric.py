# -*- coding: utf-8 -*-
"""Classification metric"""

import logging
from statistics import mean

import torch
import mlflow


log = logging.getLogger(__name__)


class ClassificationMetric:
    """Classification metric

    This modulel is metric for classification.

    Attributes:
        num_class: Number of classes.
        classes: List of class labels.
        eps: For calculation of precision, recall and f1score.
        loss_list: List of losses during 1 epoch.
        cmx: Confusion matrix during 1 epoch.
        model_score: Indicator of the excellence of model. The higher the value, the better.

    Note:
        At the beginning of every epoch, loss_list and cmx must be empty.

    """


    def __init__(self, cfg: object) -> None:
        """Initialization

        Args:
            cfg: Config.

        """

        self.num_class = cfg.data.dataset.num_class
        self.classes = cfg.data.dataset.classes
        self.eps = 1e-9

        self.loss_list = []
        self.cmx = torch.zeros(self.num_class, self.num_class, dtype=torch.int64)


    def update(self, preds, targets, loss) -> None:
        """Update loss and cmx

        Args:
            preds: Predictions.
            targets: Target values.
            loss: Loss.

        """

        stacked = torch.stack((targets, preds), dim=1)
        for p in stacked:
            tl, pl = p.tolist()
            self.cmx[tl, pl] = self.cmx[tl, pl] + 1

        self.loss_list.append(loss)

        
    def calc(self, epoch: int, mode: str) -> None:
        """Calculate metrics
        
        Calculates accuracy, loss, precision, recall and f1score.
        
        Args:
            epoch: Current epoch.
            mode: Mode. 
                train: For trainning.
                eval: For validation or test.
            
        """

        tp = torch.diag(self.cmx).to(torch.float32)
        fp = (self.cmx.sum(axis=1) - torch.diag(self.cmx)).to(torch.float32)
        fn = (self.cmx.sum(axis=0) - torch.diag(self.cmx)).to(torch.float32)

        acc = (100.0 * torch.sum(tp) / torch.sum(self.cmx)).item()
        loss = mean(self.loss_list)

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1score = tp / (tp + 0.5 * (fp + fn) + self.eps)

        if mode == "train":
            metrics = {
                "accuracy": acc,
                "loss": loss,
            }
            log.info(f"\taccuracy: {acc}")
            log.info(f"\tloss: {loss}")

        elif mode == "eval":
            metrics = {
                "val_accuracy": acc,
                "val_loss": loss,
            }
            log.info(f"\tval_accuracy: {acc}")
            log.info(f"\tval_loss: {loss}")

        if mlflow.active_run():
            mlflow.log_metrics(metrics, step = epoch)
        self.model_score = acc
        

    def reset_states(self) -> None:
        """Reset states

        Resets loss_list and cmx

        """

        self.loss_list = []
        self.cmx = torch.zeros(self.num_class, self.num_class, dtype=torch.int64)