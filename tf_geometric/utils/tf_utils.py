# coding=utf-8

import tensorflow as tf
import warnings


def tf_func_warn(func):
    warnings.warn("Using @tf_utils.function to speed-up functions is only available for TensorFlow 2.x. Upgrade your TensorFlow to 2.x and the performance can be improved dramatically with @tf_utils.function")
    return func


# disable tf.function for tf 1
if tf.__version__[0] == "1":
    function = tf_func_warn
else:
    function = tf.function

