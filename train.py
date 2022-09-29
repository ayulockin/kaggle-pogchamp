import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import wandb
from wandb.keras import WandbCallback
import tensorflow as tf

from pogchamp.data import GetDataloader
from pogchamp.model import get_model
from pogchamp import callbacks
from pogchamp import utils

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
flags.DEFINE_bool("log_model", False, "Checkpoint model while training.")
flags.DEFINE_bool("log_eval", False, "Log model prediction, needs --wandb argument as well.")

# Grow GPU memory as required.
utils.grow_gpus()

# Data rquirements
DATA_PATH = "../corn"
label2id = {
    "pure": 0,
    "broken": 1,
    "silkcut": 2,
    "discolored": 3,
}


def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    # print(config)

    CALLBACKS = []
    # Initialize a Weights and Biases run.
    if FLAGS.wandb:
        run = wandb.init(
            project=CONFIG.value.wandb_config.project,
            job_type='train',
            config=config.to_dict(),
        )
        # WandbCallback for experiment tracking
        CALLBACKS += [callbacks.WandBMetricsLogger()]

    # Load the dataframe and clean it.
    train_df = pd.read_csv(f"{DATA_PATH}/train_split.csv")[["image", "label"]]
    valid_df = pd.read_csv(f"{DATA_PATH}/valid_split.csv")[["image", "label"]]

    def apply_path(row):
        return f"{DATA_PATH}/{row.image}"
    train_df["image"] = train_df.apply(lambda row: apply_path(row), axis=1)
    valid_df["image"] = valid_df.apply(lambda row: apply_path(row), axis=1)

    def map_label_id(row):
        return label2id[row.label]
    train_df["label"] = train_df.apply(lambda row: map_label_id(row), axis=1)
    valid_df["label"] = valid_df.apply(lambda row: map_label_id(row), axis=1)

    # Get dataloader
    make_dataloader = GetDataloader(config)
    trainloader = make_dataloader.get_dataloader(train_df.image.values, train_df.label.values)
    validloader = make_dataloader.get_dataloader(valid_df.image.values, valid_df.label.values, dataloader_type="valid")

    # Get model
    tf.keras.backend.clear_session()
    model = get_model(config)
    model.summary()

    # Initialize callbacks
    callback_config = config.callback_config
    # Builtin early stopping callback
    if callback_config.use_earlystopping:
        earlystopper = callbacks.get_earlystopper(config)
        CALLBACKS += [earlystopper]
    # Built in callback to reduce learning rate on plateau
    if callback_config.use_reduce_lr_on_plateau:
        reduce_lr_on_plateau = callbacks.get_reduce_lr_on_plateau(config)
        CALLBACKS += [reduce_lr_on_plateau]

    # Initialize Model checkpointing callback
    if FLAGS.log_model:
        # Custom W&B model checkpoint callback
        model_checkpointer = callbacks.get_model_checkpoint_callback(config)
        CALLBACKS += [model_checkpointer]

    if wandb.run is not None:
        if FLAGS.log_eval:
            model_pred_viz = callbacks.get_evaluation_callback(config, validloader)
            CALLBACKS += [model_pred_viz]

    # Compile the model
    model.compile(
        optimizer = config.train_config.optimizer,
        loss = config.train_config.loss,
        metrics = config.train_config.metrics
    )

    # Train the model
    model.fit(
        trainloader,
        validation_data = validloader,
        epochs = config.train_config.epochs,
        callbacks=CALLBACKS
    )


if __name__ == "__main__":
    app.run(main)
