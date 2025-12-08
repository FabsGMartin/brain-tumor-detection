"""
Utility functions for brain tumor segmentation model.

This module contains custom functions for model architecture and loss/metrics.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    UpSampling2D,
    Concatenate,
)


def resblock(X, f):
    """
    Residual block for ResUNet architecture.

    Args:
        X: Input tensor
        f: Number of filters

    Returns:
        Output tensor after residual block
    """
    # Make a copy of input
    X_copy = X

    # Main path
    X = Conv2D(f, kernel_size=(1, 1), strides=(1, 1), kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    X = Conv2D(
        f,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer="he_normal",
    )(X)
    X = BatchNormalization()(X)

    # Short path
    X_copy = Conv2D(
        f, kernel_size=(1, 1), strides=(1, 1), kernel_initializer="he_normal"
    )(X_copy)
    X_copy = BatchNormalization()(X_copy)

    # Adding the output from main path and short path together
    X = Add()([X, X_copy])
    X = Activation("relu")(X)

    return X


def upsample_concat(x, skip):
    """
    Function to upscale and concatenate the values passed.

    Args:
        x: Input tensor to upsample
        skip: Skip connection tensor to concatenate

    Returns:
        Concatenated tensor
    """
    x = UpSampling2D((2, 2))(x)
    merge = Concatenate()([x, skip])

    return merge


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice coefficient metric for segmentation.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient value
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    """
    Dice loss for segmentation.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dice loss value
    """
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """
    Combined Binary Cross-Entropy and Dice loss.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Combined loss value
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return bce + dl


def iou_coef(y_true, y_pred, smooth=1):
    """
    Intersection over Union (IoU) coefficient metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        smooth: Smoothing factor to avoid division by zero

    Returns:
        IoU coefficient value
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)
