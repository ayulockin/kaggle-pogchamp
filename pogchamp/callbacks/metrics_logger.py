from typing import Optional

from tensorflow.keras import callbacks

import wandb


class WandBMetricsLogger(callbacks.Callback):
    """`WandbMetricsLogger` automatically logs the `logs` dictionary
    that callback methods take as argument to wandb.

    It also logs the system metrics to wandb.

    Arguments:
        log_batch_frequency (int): if None, callback will log every epoch.
            If set to integer, callback will log training metrics every `log_batch_frequency`
            batches.
    """

    def __init__(self, log_batch_frequency: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandBMetricsLogger()")

        self.log_batch_frequency = log_batch_frequency
        self.global_batch = 0

        if self.log_batch_frequency is not None:
            # define custom x axis for batch logging.
            wandb.define_metric("batch/batch_step")
            # set all other batch/ metrics to use this step
            wandb.define_metric("batch/*", step_metric="batch/batch_step")

    def on_epoch_end(self, epoch: int, logs: dict = {}):
        """Called at the end of an epoch."""
        wandb.log({"epoch": epoch}, commit=False)
        wandb.log(logs, commit=True)

    def on_batch_end(self, batch: int, logs: dict = {}):
        """A backwards compatibility alias for `on_train_batch_end`."""
        if self.log_batch_frequency and batch % self.log_batch_frequency == 0:
            wandb.log({"batch/batch_step": self.global_batch}, commit=False)

            logs = {f"batch/{k}": v for k, v in logs.items()}
            wandb.log(logs, commit=True)

            self.global_batch += self.log_batch_frequency

    def on_train_batch_end(self, batch: int, logs: dict = {}):
        """Called at the end of a training batch in `fit` methods."""
        self.on_batch_end(batch, logs)
