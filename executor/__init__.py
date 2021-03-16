# -*- coding: utf-8 -*-
"""Executor

These functions are for execution.

"""

from tqdm import tqdm
import torch
import mlflow


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

    """

    model.network.eval()
    print('\n Evaluation:')

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
    
    model.metric.result(epoch, mode='eval')

    eval_acc = model.metric.acc
    model.metric.reset_states()
    print(f'acc: {eval_acc}')

    return eval_acc


def train(model: object, train_dataloader: object, val_dataloader: object) -> None:
    """Train

    Trains model.

    Args:
        model: Model.
        train_dataloader: Dataloader for training.
        val_dataloader: Dataloader for validation.

    """

    epochs = range(model.cfg.train.epochs)

    best_acc = 0.0

    mlflow.set_tracking_uri(f"file:///workspace/mlruns")
    mlflow.set_experiment(model.cfg.experiment.name)

    with mlflow.start_run():
        for epoch in epochs:
            print(f'\n==================== Epoch: {epoch} ====================')
            print('\n Train:')
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

                    steps = epoch * len(train_dataloader) + idx
                    mlflow.log_metric("loss", loss.item(), step=steps)

            model.metric.result(epoch, mode='train')
            model.metric.reset_states()

            val_acc = eval(model=model, eval_dataloader=val_dataloader, epoch=epoch)
            model.metric.reset_states()

            # save best ckpt
            if val_acc > best_acc:
                best_acc = val_acc
                save_ckpt(model=model, epoch=epoch)
    

