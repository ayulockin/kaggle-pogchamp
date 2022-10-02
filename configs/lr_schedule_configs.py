import os

import ml_collections
from ml_collections import config_dict


def warmup_cosine_decay_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.start_lr = 0.0
    configs.target_lr = 1e-3
    configs.hold = 0
    configs.total_steps = config_dict.placeholder(int)
    configs.warmup_steps = config_dict.placeholder(int)

    return configs
