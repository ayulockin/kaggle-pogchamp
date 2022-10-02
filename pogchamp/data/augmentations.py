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


def get_mixup(args):
    return keras_cv.layers.MixUp(alpha=args.alpha)


def get_augmix(args):
    return keras_cv.layers.AugMix(
        value_range=args.value_range,
        severity=args.severity,
        num_chains=args.num_chains,
        chain_depth=args.chain_depth,
        alpha=args.alpha,
    )


def get_random_flip(args):
    return tf.keras.layers.RandomFlip(mode=args.mode)


def get_random_rotation(args):
    return tf.keras.layers.RandomRotation(
        factor=args.factor,
        fill_mode=args.fill_mode,
        interpolation=args.interpolation,
    )


def get_random_zoom(args):
    return tf.keras.layers.RandomZoom(
        height_factor=args.height_factor,
        width_factor=args.width_factor,
        fill_mode=args.fill_mode,
        interpolation=args.interpolation,
    )
