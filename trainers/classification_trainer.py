# -*- coding: utf-8 -*-
"""Classification Trainer"""

import logging

from tqdm import tqdm
import torch
import mlflow
from omegaconf import DictConfig

from trainers.base_trainer import BaseTrainer
from models import get_model
from data import get_dataloader
from trainers.metrics import get_metrics


log = logging.getLogger(__name__)


class ClassificationTrainer(BaseTrainer):
    """ClassificationTrainer
    
    Attributes:
        cfg: Config of project.
        model: Model.
    
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialization
    
        Args:
            cfg: Config of project.

        """

        super().__init__(cfg)
        self.model = get_model(self.cfg)
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.metrics = get_metrics(self.cfg)


    def execute(self, eval: bool) -> None:
        """Execution

        Execute train or eval.

        Args:
            eval: For evaluation mode.
                True: Execute eval.
                False: Execute train.

        """

        if not eval:
            self.train_dataloader, self.val_dataloader = get_dataloader(self.cfg, mode="trainval")
            self.train()

        else:
            self.test_dataloader = get_dataloader(self.cfg, mode="test")
            self.eval()


    def train(self) -> None:
        """Train

        Trains model.

        """

        super().train()

        epochs = range(self.cfg.train.epochs)

        with mlflow.start_run():
            self.log_params()

            for epoch in epochs:
                log.info(f"==================== Epoch: {epoch} ====================")
                log.info(f"Train:")
                self.model.network.train()

                with tqdm(self.train_dataloader, ncols=100) as pbar:
                    for idx, (inputs, targets) in enumerate(pbar):
                        inputs = inputs.to(self.model.device)
                        targets = targets.to(self.model.device)
                        outputs = self.model.network(inputs)

                        loss = self.model.criterion(outputs, targets)

                        loss.backward()

                        self.model.optimizer.step()
                        self.model.optimizer.zero_grad()

                        self.metrics.batch_update(outputs=outputs.cpu().detach().clone(),
                                            targets=targets.cpu().detach().clone(),
                                            loss=loss.item())

                        pbar.set_description(f'train epoch:{epoch}')

                self.metrics.epoch_update(epoch, mode='train')
                self.eval(eval_dataloader=self.val_dataloader, epoch=epoch)

                if self.metrics.judge_update_ckpt:
                    self.model.save_ckpt(epoch=epoch, ckpt_path=self.cfg.train.ckpt_path)
                    log.info("Saved the check point.")

            log.info("Successfully trained the model.")

            self.log_artifacts()


    def eval(self, eval_dataloader: object = None, epoch: int = 0) -> float:
        """Evaluation

        Evaluates model.

        Args:
            eval_dataloader: Dataloader.
            epoch: Number of epoch.

        Returns:
            model_score: Indicator of the excellence of model. The higher the value, the better.

        """
        
        super().eval()

        if not eval_dataloader:
            eval_dataloader = self.test_dataloader

        self.model.network.eval()

        with torch.no_grad():
            with tqdm(eval_dataloader, ncols=100) as pbar:
                for idx, (inputs, targets) in enumerate(pbar):
                    inputs = inputs.to(self.model.device)
                    targets = targets.to(self.model.device)

                    outputs = self.model.network(inputs)

                    loss = self.model.criterion(outputs, targets)
                    self.model.optimizer.zero_grad()

                    self.metrics.batch_update(outputs=outputs.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=loss.item())

                    pbar.set_description(f'eval epoch: {epoch}')
        
        self.metrics.epoch_update(epoch, mode='eval')

        return self.metrics.model_score