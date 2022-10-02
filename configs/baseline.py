import os

import ml_collections
from ml_collections import config_dict

from configs.augmentation_configs import *


def get_wandb_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.project = "pogchamp"
    configs.log_data_type = "train"
    configs.log_num_samples = -1  # passing -1 will upload the complete dataset
    configs.log_evaluation_table = False
    # configs.entity = "wandb_fc"

    return configs


def get_dataset_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.image_height = 96
    configs.image_width = 96
    configs.channels = 3
    configs.shuffle_buffer = 1024
    configs.batch_size = 64
    configs.num_classes = 4
    configs.do_cache = False
    configs.use_augmentations = True
    # Always True since images are of varying sizes.
    configs.apply_resize = True
    configs.apply_one_hot = True

    return configs


def get_augmentation_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict
    configs.randaugment = randaugment_config()
    configs.use_augmentations = ["randaugment"]

    return configs


def get_model_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.model_img_height = 96
    configs.model_img_width = 96
    configs.model_img_channels = 3
    configs.backbone = "resnet50"
    configs.use_pretrained_weights = True
    configs.dropout_rate = 0.5
    configs.post_gap_dropout = False

    return configs


def get_callback_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    # Early stopping
    configs.use_earlystopping = True
    configs.early_patience = 6
    # Reduce LR on plateau
    configs.use_reduce_lr_on_plateau = True
    configs.rlrp_factor = 0.2
    configs.rlrp_patience = 3
    # Model checkpointing
    configs.checkpoint_filepath = "wandb/model_{epoch}"
    configs.save_best_only = True
    # Model Prediction Viz
    configs.viz_num_images = 100

    return configs


def get_train_configs() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.epochs = 15
    configs.use_augmentations = False
    configs.use_class_weights = False
    configs.optimizer = "adam"
    configs.sgd_momentum = 0.9
    configs.loss = "categorical_crossentropy"
    configs.metrics = ["accuracy"]

    return configs


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.wandb_config = get_wandb_configs()
    config.dataset_config = get_dataset_configs()
    config.aug_config = get_augmentation_configs()
    config.model_config = get_model_configs()
    config.callback_config = get_callback_configs()
    config.train_config = get_train_configs()

    return config
