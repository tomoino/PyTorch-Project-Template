# -*- coding: utf-8 -*-
"""Load files

This module is for loading files.

"""

import yaml


def load_yaml(cfg_path: str) -> dict:
    """Loads yaml
    
    Loads config yaml

    Args:
        cfg_path: Path of config file.
    
    Returns:
        Configs.

    """

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
        
    return cfg