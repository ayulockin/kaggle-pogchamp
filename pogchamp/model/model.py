import json
import os
import tempfile

import ml_collections
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import wandb


def get_convnext_model(model_name, weights):
    variant = model_name.split("-")[-1]
    if variant == "b":
        backbone = tf.keras.applications.convnext.ConvNeXtBase(
            include_top=False,
            weights=weights
        )
    if variant == "s":
        backbone = tf.keras.applications.convnext.ConvNeXtSmall(
            include_top=False,
            weights=weights
        )
    if variant == "t":
        backbone = tf.keras.applications.convnext.ConvNeXtTiny(
            include_top=False,
            weights=weights
        )
    
    return backbone


def get_effnetv2_backbone(model_name, weights):
    variant = model_name.split("-")[-1]
    if variant == "b2":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(
            include_top=False,
            weights=weights
        )
    if variant == "b0":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
            include_top=False,
            weights=weights
        )
    if variant == "s":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False,
            weights=weights
        )

    return backbone


def get_backbone(args):
    """Get backbone for the model.

    Args:
        args (ml_collections.ConfigDict): Configuration.
    """
    weights = None
    if args.model_config.use_pretrained_weights:
        weights = "imagenet"

    if args.model_config.backbone == "vgg16":
        base_model = tf.keras.applications.VGG16(include_top=False, weights=weights)
        base_model.trainable = True
    elif args.model_config.backbone == "resnet50":
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=weights)
        base_model.trainable = True
    elif "convnext" in args.model_config.backbone:
        base_model = get_convnext_model(args.model_config.backbone, weights)
        base_model.trainable = True
    elif "effnetv2" in args.model_config.backbone:
        base_model = get_effnetv2_backbone(args.model_config.backbone, weights)
        base_model.trainable = True
    else:
        raise NotImplementedError("Not implemented for this backbone.")

    return base_model


def get_model(args):
    """Get an image classifier with a CNN based backbone.

    Args:
        args (ml_collections.ConfigDict): Configuration.
    """
    # Backbone
    base_model = get_backbone(args)

    # Stack layers
    inputs = layers.Input(
        shape=(
            args.model_config.model_img_height,
            args.model_config.model_img_width,
            args.model_config.model_img_channels,
        )
    )

    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    if args.model_config.post_gap_dropout:
        x = layers.Dropout(args.model_config.dropout_rate)(x)
    outputs = layers.Dense(args.dataset_config.num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)
