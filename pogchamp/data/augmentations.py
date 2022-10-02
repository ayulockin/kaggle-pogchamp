import keras_cv
import tensorflow as tf


def get_randaugmment(args):
    """
    Get RandAugment augmentation policy.

    Input:
        args: ML collection config_dict at the nested level
            such that the arguments can be accessed.

    """
    return keras_cv.layers.RandAugment(
        value_range=args.value_range,
        augmentations_per_image=args.augmentations_per_image,
        magnitude=args.magnitude,
        magnitude_stddev=args.magnitude_stddev,
        rate=args.rate,
    )
