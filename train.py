# -*- coding: utf-8 -*-
"""Main script

This module is for training and evaluation.

Args:
    project (str): Name of project yaml file.
    eval (bool): For evaluation mode.

"""

import hydra
from omegaconf import DictConfig

from trainers import get_trainer


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function
    
    Builds model, loads data, trains and evaluates

    Args:
        cfg: Config.
    
    Returns:
        None.

    """

    trainer = get_trainer(cfg.project)
    trainer.execute(eval=cfg.eval)


if __name__ == '__main__':
    main()