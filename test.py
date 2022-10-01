import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags
from ml_collections.config_flags import config_flags

import wandb
from wandb.keras import WandbCallback

from pogchamp import callbacks, utils
from pogchamp.data import GetDataloader, preprocess_dataframe
from pogchamp.model import get_model, download_model

# Config
FLAGS = flags.FLAGS
CONFIG = config_flags.DEFINE_config_file("config")
flags.DEFINE_string(
    "model_artifact_path", None, "Model checkpoint saved as W&B artifact."
)
flags.mark_flag_as_required("model_artifact_path")
flags.DEFINE_bool("wandb", False, "MLOps pipeline for our classifier.")
flags.DEFINE_bool(
    "log_eval", False, "Log model prediction, needs --wandb argument as well."
)

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
id2label = {v: k for k, v in label2id.items()}


def main(_):
    # Get configs from the config file.
    config = CONFIG.value
    # print(config)

    CALLBACKS = []
    # Initialize a Weights and Biases run.
    if FLAGS.wandb:
        run = wandb.init(
            project=CONFIG.value.wandb_config.project,
            job_type="train",
            config=config.to_dict(),
        )
        # WandbCallback for experiment tracking
        CALLBACKS += [callbacks.WandBMetricsLogger()]

    # Load the dataframe and clean it.
    df = pd.read_csv(f"{DATA_PATH}/test.csv")[["image"]]
    df = preprocess_dataframe(df)
    img_paths= df.image.values

    # Prepare dataloader.
    make_dataloader = GetDataloader(config)
    dataloader = make_dataloader.get_dataloader(
        img_paths, dataloader_type="test"
    )

    # Download the model and load it.
    model_path = download_model(FLAGS.model_artifact_path)
    if wandb.run is not None:
        artifact = run.use_artifact(FLAGS.model_artifact_path, type='model')
    print("Path to the model checkpoint: ", model_path)
    
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Test the model
    predictions = model.predict(dataloader)
    predictions = np.argmax(predictions, axis=1)

    # Write csv file
    pred_df = pd.read_csv(f"{DATA_PATH}/test.csv")
    pred_df["label"] = predictions

    def apply_id2label(row):
        return id2label[row.label]
    pred_df["label"] = pred_df.apply(lambda row: apply_id2label(row), axis=1)
    pred_df = pred_df[["seed_id", "label"]]
    pred_df.to_csv('submission.csv',index=False)
    


if __name__ == "__main__":
    app.run(main)
