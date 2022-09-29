import tensorflow as tf

import wandb
from pogchamp.utils import BaseWandbEvalCallback


class WandbClfCallback(BaseWandbEvalCallback):
    def __init__(self, args, dataloader, is_train=True):
        data_table_columns = ["idx", "image", "ground_truth"]
        pred_table_columns = ["epoch"] + data_table_columns + ["prediction"]
        super().__init__(data_table_columns, pred_table_columns, is_train)

        self.args = args
        # Make unbatched iterator from `tf.data.Dataset`.
        self.val_ds = dataloader.unbatch().take(self.args.callback_config.viz_num_images)

    def add_ground_truth(self, logs):
        for idx, (image, label) in enumerate(self.val_ds.as_numpy_iterator()):
            if self.args.dataset_config.apply_one_hot:
                label = tf.argmax(label, axis=-1)
            self.data_table.add_data(idx, wandb.Image(image), label)

    def add_model_predictions(self, epoch, logs):
        data_table_ref = self.data_table_ref
        table_idxs = data_table_ref.get_index()

        for idx, (image, label) in enumerate(self.val_ds.as_numpy_iterator()):
            pred = self.model.predict(tf.expand_dims(image, axis=0), verbose=0)
            pred = tf.squeeze(tf.argmax(pred, axis=-1), axis=0)

            self.pred_table.add_data(
                epoch,
                data_table_ref.data[idx][0],
                data_table_ref.data[idx][1],
                data_table_ref.data[idx][2],
                pred,
            )


def get_evaluation_callback(args, dataloader, is_train=True):
    return WandbClfCallback(
        args,
        dataloader,
        is_train=is_train,
    )
