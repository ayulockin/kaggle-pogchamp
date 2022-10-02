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
    configs.alpha=0.6 

    return configs


def random_zoom_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.height_factor = 0.1
    configs.width_factor = None
    configs.fill_mode = 'reflect'
    configs.interpolation = 'bilinear'

    return configs


def random_rotation_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.factor = 0.1
    configs.fill_mode='reflect'
    configs.interpolation='bilinear'

    return configs


def random_flip() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.mode = "horizontal_and_vertical"

    return configs
