import os

import ml_collections


def randaugment_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.value_range = (0.0, 255.0)
    configs.augmentations_per_image = 3
    configs.magnitude = 0.5
    configs.magnitude_stddev = 0.15
    configs.rate = 0.9090909090909091
    configs.geometric = True

    return configs


def mixup_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.alpha = 0.2

    return configs

def augmix_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.value_range = (0., 255.)
    configs.severity=0.3
    configs.num_chains=3
    configs.chain_depth=[1, 3]
    configs.alpha=1.0   

    return configs