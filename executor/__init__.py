# -*- coding: utf-8 -*-
"""Executor

These functions are for execution.

"""

import torch
from tqdm import tqdm


def save_ckpt(cfg: dict, model: object, epoch: int) -> None:
    """Save checkpoint

    Saves checkpoint.

    Args:
        model: Model.
        dataloader: Dataloader.
        epoch: Number of epoch.

    """

    ckpt_path = f"{cfg['train']['logdir']}/ckpt/best_acc_ckpt.pth"

    torch.save({
        'epoch': epoch,
        'model': model.model,
        'model_state_dict': model.model.state_dict(),
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

    model.model.eval()
    print('\n Evaluation:')

    with torch.no_grad():
        with tqdm(eval_dataloader, ncols=100) as pbar:
            for idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(model.device)
                targets = targets.to(model.device)

                outputs = model.model(inputs)

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

    return eval_acc


def train(config: dict, model: object, train_dataloader: object, val_dataloader: object) -> None:
    """Train

    Trains model.

    Args:
        config: Config.
        model: Model.
        train_dataloader: Dataloader for training.
        val_dataloader: Dataloader for validation.

    """

    epochs = range(config["train"]["epochs"])
    model.build()

    best_acc = 0.0

    for epoch in epochs:
        print(f'\n==================== Epoch: {epoch} ====================')
        print('\n Train:')
        model.model.train()

        with tqdm(train_dataloader, ncols=100) as pbar:
            for idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(model.device)
                targets = targets.to(model.device)

                outputs = model.model(inputs)

                loss = model.criterion(outputs, targets)

                loss.backward()

                model.optimizer.step()
                model.optimizer.zero_grad()

                preds = outputs.argmax(axis=1)
                model.metric.update(preds=preds.cpu().detach().clone(),
                                    targets=targets.cpu().detach().clone(),
                                    loss=loss.item())

                pbar.set_description(f'train epoch:{epoch}')

        model.metric.result(epoch, mode='train')
        model.metric.reset_states()

        val_acc = eval(cfg, model=model, eval_dataloader=val_dataloader)
        model.metric.reset_states()

        # save best ckpt
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_ckpt(cfg, model=model, epoch=epoch)
    

