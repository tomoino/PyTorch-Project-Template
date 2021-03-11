# -*- coding: utf-8 -*-
"""Classification metric

This modulel is metric for classification.

"""

import csv
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch


class ClassificationMetric:
    def __init__(self, cfg: dict):
        self.class_num = cfg["train"]["class_num"]
        self.classes = cfg["train"]["classes"]
        self.metric_dir = Path(f"{cfg['train']['logdir']}/metric")
        self.eps = 1e-9

        self.loss_list = []
        self.cmx = torch.zeros(self.class_num, self.class_num, dtype=torch.int64)

    def update(self, preds, targets, loss) -> None:
        stacked = torch.stack((targets, preds), dim=1)
        for p in stacked:
            tl, pl = p.tolist()
            self.cmx[tl, pl] = self.cmx[tl, pl] + 1

        self.loss_list.append(loss)
        
    def result(self, epoch: int, mode: str):
        """Metric(acc, loss, precision, recall, f1score), Logging, Save and Plot CMX
        
        Args:
            epoch: Current epoch
            mode: Mode. 
                train: For trainning.
                eval: For validation or test.
            
        """
        tp = torch.diag(self.cmx).to(torch.float32)
        fp = (self.cmx.sum(axis=1) - torch.diag(self.cmx)).to(torch.float32)
        fn = (self.cmx.sum(axis=0) - torch.diag(self.cmx)).to(torch.float32)

        self.acc = (100.0 * torch.sum(tp) / torch.sum(self.cmx)).item()
        self.loss = mean(self.loss_list)

        self.precision = tp / (tp + fp + self.eps)
        self.recall = tp / (tp + fn + self.eps)
        self.f1score = tp / (tp + 0.5 * (fp + fn) + self.eps) # micro f1score

    def reset_states(self):
        self.loss_list = []
        self.cmx = torch.zeros(self.class_num, self.class_num, dtype=torch.int64)
