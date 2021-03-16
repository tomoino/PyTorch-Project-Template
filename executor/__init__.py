# -*- coding: utf-8 -*-
"""Executor

These functions are for execution.

"""

import logging

from tqdm import tqdm
import torch
import mlflow


log = logging.getLogger(__name__)


def save_ckpt(model: object, epoch: int) -> None:
    """Save checkpoint

    Saves checkpoint.

    Args:
        model: Model.
        epoch: Number of epoch.

    """

    ckpt_path = "./best_acc_ckpt.pth"

    torch.save({
        'epoch': epoch,
        'model': model.network,
        'model_state_dict': model.network.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
    }, ckpt_path)


def eval(model: object, eval_dataloader: object, epoch: int = 0) -> float:
    """Evaluation

    Evaluates model.

    Args:
        model: Model.
        eval_dataloader: Dataloader.
        epoch: Number of epoch.

    Returns:
        model_score: Indicator of the excellence of model. The higher the value, the better.

    """

    model.network.eval()
    log.info('\n Evaluation:')

    with torch.no_grad():
        with tqdm(eval_dataloader, ncols=100) as pbar:
            for idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(model.device)
                targets = targets.to(model.device)

                outputs = model.network(inputs)

                loss = model.criterion(outputs, targets)
                model.optimizer.zero_grad()

                preds = outputs.argmax(axis=1)
                model.metric.update(preds=preds.cpu().detach().clone(),
                                    targets=targets.cpu().detach().clone(),
                                    loss=loss.item())

                pbar.set_description(f'eval epoch: {epoch}')
    
    model.metric.calc(epoch, mode='eval')
    model.metric.reset_states()

    return model.metric.model_score


def log_param(cfg: object) -> None:
    """Log parameters

    Args:
        cfg: Config.

    """

    params = {
        "dataset": cfg.data.dataset.name,
        "model": cfg.model.name,
        "batch_size": cfg.train.batch_size,
        "epochs": cfg.train.epochs,
        "lr": cfg.train.optimizer.lr
    }

    mlflow.log_params(params)


def train(model: object, train_dataloader: object, val_dataloader: object) -> None:
    """Train

    Trains model.

    Args:
        model: Model.
        train_dataloader: Dataloader for training.
        val_dataloader: Dataloader for validation.

    """

    epochs = range(model.cfg.train.epochs)

    best_score = 0.0

    mlflow.set_tracking_uri("file:///workspace/mlruns")
    mlflow.set_experiment(model.cfg.experiment.name)

    with mlflow.start_run():
        log_param(model.cfg)
        for epoch in epochs:
            log.info(f'\n==================== Epoch: {epoch} ====================')
            log.info('\n Train:')
            model.network.train()

            with tqdm(train_dataloader, ncols=100) as pbar:
                for idx, (inputs, targets) in enumerate(pbar):
                    inputs = inputs.to(model.device)
                    targets = targets.to(model.device)
                    outputs = model.network(inputs)

                    loss = model.criterion(outputs, targets)

                    loss.backward()

                    model.optimizer.step()
                    model.optimizer.zero_grad()

                    preds = outputs.argmax(axis=1)
                    model.metric.update(preds=preds.cpu().detach().clone(),
                                        targets=targets.cpu().detach().clone(),
                                        loss=loss.item())

                    pbar.set_description(f'train epoch:{epoch}')

            model.metric.calc(epoch, mode='train')
            model.metric.reset_states()

            model_score = eval(model=model, eval_dataloader=val_dataloader, epoch=epoch)
            model.metric.reset_states()

            if model_score > best_score:
                best_score = model_score
                save_ckpt(model=model, epoch=epoch)

        mlflow.log_artifact("train.log")
        mlflow.log_artifact(".hydra/config.yaml")