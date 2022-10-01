import tensorflow as tf


def grow_gpus():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def initialize_device():
    return (
        tf.distribute.MirroredStrategy()
        if len(tf.config.list_physical_devices("GPU")) > 1
        else tf.distribute.OneDeviceStrategy(device="/gpu:0")
    )
