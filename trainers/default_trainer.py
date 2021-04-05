# -*- coding: utf-8 -*-
"""Default Trainer"""

import logging

from tqdm import tqdm
import torch
import mlflow

from trainers.base_trainer import BaseTrainer
from models import get_model
from data import get_dataloader


log = logging.getLogger(__name__)


class DefaultTrainer(BaseTrainer):
    """DefaultTrainer
    
    Attributes:
        cfg: Config of project.
        model: Model.
    
    """

    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Args:
            cfg: Config of project.

        """

        super().__init__(cfg)
        self.model = get_model(self.cfg)
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None


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

        epochs = range(self.model.cfg.train.epochs)
        best_score = 0.0

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

                        preds = outputs.argmax(axis=1)
                        self.model.metric.update(preds=preds.cpu().detach().clone(),
                                            targets=targets.cpu().detach().clone(),
                                            loss=loss.item())

                        pbar.set_description(f'train epoch:{epoch}')

                self.model.metric.calc(epoch, mode='train')
                self.model.metric.reset_states()

                model_score = self.eval(eval_dataloader=self.val_dataloader, epoch=epoch)
                self.model.metric.reset_states()

                if model_score > best_score:
                    best_score = model_score
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

                    preds = outputs.argmax(axis=1)
                    self.model.metric.update(preds=preds.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=loss.item())

                    pbar.set_description(f'eval epoch: {epoch}')
        
        self.model.metric.calc(epoch, mode='eval')
        self.model.metric.reset_states()

        return self.model.metric.model_score