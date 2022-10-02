import numpy as np
import tensorflow as tf


def lr_warmup_cosine_decay(
    global_step, warmup_steps, hold=0, total_steps=0, start_lr=0.0, target_lr=1e-3
):
    # Cosine decay
    # There is no tf.pi so we wrap np.pi as a TF constant
    learning_rate = (
        0.5
        * target_lr
        * (
            1
            + tf.cos(
                tf.constant(np.pi)
                * (global_step - warmup_steps - hold)
                / float(total_steps - warmup_steps - hold)
            )
        )
    )

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = tf.where(
            global_step > warmup_steps + hold, learning_rate, target_lr
        )

    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmupCosineDecayCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0
    ):

        super(WarmupCosineDecayCallback, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(
            global_step=self.global_step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            hold=self.hold,
        )
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


def get_warmup_cosine_decay_callback(args):
    args = args.lr_config.warmup_cosine_decay
    schedule = WarmupCosineDecayCallback(
        start_lr=args.start_lr,
        target_lr=args.target_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        hold=args.warmup_steps,
    )

    return schedule
