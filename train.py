# -*- coding: utf-8 -*-
"""Main script

This module is for training and evaluation.

Args:
    project (str): Name of project yaml file.
    eval (bool): For evaluation mode.

"""

import hydra
from omegaconf import DictConfig

from models import get_model
from data import get_dataloader
from executor import train, eval


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function
    
    Builds model, loads data, trains and evaluates

    Args:
        cfg: Config.
    
    Returns:
        None.

    """

    model = get_model(cfg.project)

    if not cfg.eval:
        train_dataloader, val_dataloader = get_dataloader(cfg.project, mode="trainval")
        train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    else:
        test_dataloader = get_dataloader(cfg.project, mode="test")
        eval(model=model, eval_dataloader=test_dataloader)


if __name__ == '__main__':
    main()