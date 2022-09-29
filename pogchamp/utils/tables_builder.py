import abc
from typing import Dict, List

from tensorflow.keras.callbacks import Callback

import wandb


class BaseWandbEvalCallback(Callback, metaclass=abc.ABCMeta):
    """Base class to build Keras callbacks for model prediction visualization.

    You can build callbacks for visualizing model predictions `on_epoch_end`
    that can be passed to `model.fit()` for classification, object detection,
    segmentation, etc. tasks.

    To use this, inherit this base callback and implement the `add_ground_truth`
    and `add_model_prediction` methods.

    The base class will take care of the following:
    - Initialize `data_table` for logging ground truth and
        `pred_table` for predictions.
    - The data uploaded to `data_table` is used as reference for the
        `pred_table`. This is to reduce memory footprint. The `data_table_ref`
        is a list that can be used to access the referenced data.
        Check out the example below to see how it's done.
    - Log the tables to W&B as W&B artifacts.
    - Each new `pred_table` is logged as a new version with aliases.

    Example:
        ```
        class WandbClfEvalCallback(WandbEvalCallback):
            def __init__(self,
                        validation_data,
                        data_table_columns,
                        pred_table_columns
                    ):
                super().__init__(
                    data_table_columns,
                    pred_table_columns
                )

                self.x = validation_data[0]
                self.y = validation_data[1]

            def add_ground_truth(self):
                for idx, (image, label) in enumerate(zip(self.x, self.y)):
                    self.data_table.add_data(
                        idx,
                        wandb.Image(image),
                        label
                    )

            def add_model_predictions(self, epoch):
                preds = self.model.predict(self.x, verbose=0)
                preds = tf.argmax(preds, axis=-1)

                data_table_ref = self.data_table_ref
                table_idxs = data_table_ref.get_index()

                for idx in table_idxs:
                    pred = preds[idx]
                    self.pred_table.add_data(
                        epoch,
                        data_table_ref.data[idx][0],
                        data_table_ref.data[idx][1],
                        data_table_ref.data[idx][2],
                        pred
                    )

        model.fit(
            x,
            y,
            epochs=2,
            validation_data=(x, y),
            callbacks=[
                WandbClfEvalCallback(
                    validation_data=(x, y),
                    data_table_columns=["idx", "image", "label"],
                    pred_table_columns=["epoch", "idx", "image", "label", "pred"])
            ],
        )
        ```

    To have more fine-grained control, you can override the `on_train_begin` and
    `on_epoch_end` methods. If you want to log the samples after N batched, you
    can implement `on_train_batch_end` method.
    """

    def __init__(
        self,
        data_table_columns=List[str],
        pred_table_columns=List[str],
        is_train: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if wandb.run is None:
            raise wandb.Error(
                "You must call wandb.init() before BaseWandbEvalCallback()"
            )

        self.data_table_columns = data_table_columns
        self.pred_table_columns = pred_table_columns
        self.is_train = is_train

    def on_train_begin(self, logs=None):
        if self.is_train:
            # Initialize the data_table
            self.init_data_table(column_names=self.data_table_columns)
            # Log the ground truth data
            self.add_ground_truth(logs)
            # Log the data_table as W&B Artifacts
            self.log_data_table()

    def on_epoch_end(self, epoch, logs=None):
        if self.is_train:
            # Initialize the pred_table
            self.init_pred_table(column_names=self.pred_table_columns)
            # Log the model prediction
            self.add_model_predictions(epoch, logs)
            # Log the pred_table as W&B Artifacts
            self.log_pred_table()

    def on_test_begin(self, logs=None):
        if not self.is_train:
            # Initialize the data_table
            self.init_data_table(column_names=self.data_table_columns)
            # Log the ground truth data
            self.add_ground_truth(logs)
            # Log the data_table as W&B Artifacts
            self.log_data_table(name="test_data", table_name="test_table")

    def on_test_end(self, logs=None):
        if not self.is_train:
            # Initialize the pred_table
            self.init_pred_table(column_names=self.pred_table_columns)
            # Log the model prediction
            self.add_model_predictions(None, logs)
            # Log the pred_table as W&B Artifacts
            self.log_pred_table(type="evaluation", table_name="test_pred")

    @abc.abstractmethod
    def add_ground_truth(self, logs: Dict[str, float] = {}):
        """Use this method to write the logic for adding validation/training
        data to `data_table` initialized using `init_data_table` method.
        Example:
            ```
            for idx, data in enumerate(dataloader):
                self.data_table.add_data(
                    idx,
                    data
                )
            ```
        This method is called once `on_train_begin` or equivalent hook.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.add_ground_truth")

    @abc.abstractmethod
    def add_model_predictions(self, epoch: int, logs: Dict[str, float] = {}):
        """Use this method to write the logic for adding model prediction for
        validation/training data to `pred_table` initialized using
        `init_pred_table` method.
        Example:
            ```
            # Assuming the dataloader is not shuffling the samples.
            for idx, data in enumerate(dataloader):
                preds = model.predict(data)
                self.pred_table.add_data(
                    self.data_table_ref.data[idx][0],
                    self.data_table_ref.data[idx][1],
                    preds
                )
            ```
        This method is called `on_epoch_end` or equivalent hook.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.add_model_predictions")

    def init_data_table(self, column_names: List[str]):
        """Initialize the W&B Tables for validation data.
        Call this method `on_train_begin` or equivalent hook. This is followed by
        adding data to the table row or column wise.
        Args:
            column_names (list): Column names for W&B Tables.
        """
        self.data_table = wandb.Table(columns=column_names, allow_mixed_types=True)

    def init_pred_table(self, column_names: List[str]):
        """Initialize the W&B Tables for model evaluation.
        Call this method `on_epoch_end` or equivalent hook. This is followed by
        adding data to the table row or column wise.
        Args:
            column_names (list): Column names for W&B Tables.
        """
        self.pred_table = wandb.Table(columns=column_names)

    def log_data_table(
        self,
        name: str = "val_data",
        type: str = "eval_dataset",
        table_name: str = "val_data",
    ):
        """Log the `data_table` as W&B artifact and call
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded data (images, text, scalar, etc.).
        This allows the data to be uploaded just once.
        Args:
            name (str):  A human-readable name for this artifact, which is how
                you can identify this artifact in the UI or reference
                it in use_artifact calls. (default is 'val')
            type (str): The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'val_data')
            table_name (str): The name of the table as will be displayed in the UI.
        """
        data_artifact = wandb.Artifact(name, type=type)
        data_artifact.add(self.data_table, table_name)

        # Calling `use_artifact` uploads the data to W&B.
        wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        # We get the reference table.
        self.data_table_ref = data_artifact.get(table_name)

    def log_pred_table(
        self,
        type: str = "validation",
        table_name: str = "eval_data",
        aliases: List[str] = ["latest"],
    ):
        """Log the W&B Tables for model evaluation.
        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        Args:
            type (str): The type of the artifact, which is used to organize and
                differentiate artifacts. (default is 'val_data')
            table_name (str): The name of the table as will be displayed in the UI.
            aliases (List[str]): List of aliases for the pediction table.
        """
        pred_artifact = wandb.Artifact(f"run_{wandb.run.id}_pred", type=type)
        pred_artifact.add(self.pred_table, table_name)
        wandb.run.log_artifact(pred_artifact, aliases=aliases)
