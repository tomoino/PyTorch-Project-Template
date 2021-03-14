# -*- coding: utf-8 -*-
"""Main script

This module is for training and evaluation.

Args:
    --configfile (str): Config file.
    --eval (bool): For evaluation mode.

"""

import argparse

from utils.load import load_yaml
from data import get_dataloader
from models import get_model
from executor import train, eval


def parser() -> object:
    """Arguments parser

    Returns:
        Arguments.

    """

    parser = argparse.ArgumentParser(description='for training and evaluation')
    parser.add_argument('--configfile', type=str, default='./configs/default.yml', help='config file')
    parser.add_argument('--eval', action='store_true', help='run in evaluation mode')
    args = parser.parse_args()

    return args


def main(args) -> None:
    """Main function
    
    Builds model, loads data, trains and evaluates

    Args:
        args: Arguments.
    
    Returns:
        None.

    """

    config: dict = load_yaml(args.configfile)
    model = get_model(config)

    if not args.eval:
        train_dataloader, val_dataloader = get_dataloader(config, mode="trainval")
        train(config, model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    else:
        test_dataloader = get_dataloader(config, mode="test")
        eval(model=model, eval_dataloader=test_dataloader)

if __name__ == '__main__':
    args = parser()
    main(args)
