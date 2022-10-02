import os

import ml_collections


def randaugment_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.value_range = (0.0, 1.0)
    configs.augmentations_per_image = 3
    configs.magnitude = (0.5,)
    configs.magnitude_stddev = (0.15,)
    configs.rate = (0.9090909090909091,)
    configs.geometric = (True,)

    return configs
