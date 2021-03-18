# -*- coding: utf-8 -*-
"""Abstract base model"""

import logging
from abc import ABC

import torch
import mlflow


log = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer

    This model is inherited by all trainers.

    Attributes:
        cfg: Config of project.

    """

    def __init__(self, cfg: object) -> None:
        """Initialization

        Args:
            cfg: Config of project.

        """

        self.cfg = cfg


    def execute(self, eval: bool) -> None:
        """Execution

        Execute train or eval.

        Args:
            eval: For evaluation mode.
                True: Execute eval.
                False: Execute train.

        """
        pass


    def train(self) -> None:
        """Train

        Trains model.

        """

        log.info("Training process has begun.")
        
        mlflow.set_tracking_uri("file:///workspace/mlruns")
        mlflow.set_experiment(self.cfg.experiment.name)


    def eval(self,eval_dataloader: object = None, epoch: int = 0) -> float:
        """Evaluation

        Evaluates model.

        Args:
            eval_dataloader: Dataloader.
            epoch: Number of epoch.

        Returns:
            model_score: Indicator of the excellence of model. The higher the value, the better.

        """
        
        log.info('Evaluation:')


    def log_params(self) -> None:
        """Log parameters"""

        params = {
            "dataset": self.cfg.data.dataset.name,
            "model": self.cfg.model.name,
            "batch_size": self.cfg.train.batch_size,
            "epochs": self.cfg.train.epochs,
            "criterion": self.cfg.train.criterion.name,
            "optimizer": self.cfg.train.optimizer.name,
            "lr": self.cfg.train.optimizer.lr
        }

        mlflow.log_params(params)


    def log_artifacts(self) -> None:
        """log artifacts"""
        
        artifacts_dir = mlflow.get_artifact_uri()
        ckpt_path = f"{artifacts_dir.replace('file://','')}/{self.cfg.train.ckpt_path}"
        log.info("You can evaluate the model by running the following code.")
        log.info(f"$ python train.py eval=True project.model.initial_ckpt={ckpt_path}")

        mlflow.log_artifact("train.log")
        mlflow.log_artifact(".hydra/config.yaml")
        mlflow.log_artifact(self.cfg.train.ckpt_path)