# coding=utf-8

import tensorflow as tf
import warnings


def warn_tf1():
    warnings.warn("Using @tf_utils.function to speed-up functions is only available for TensorFlow 2.x. "
                  "Upgrade your TensorFlow to 2.x and the performance can be improved dramatically with @tf_utils.function")


def tf_func_warn(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        func = args[0]
        warn_tf1()
        return func
    else:
        def decorate(func):
            warn_tf1()
            return func

        return decorate


# disable tf.function for tf 1
if tf.__version__[0] == "1":
    function = tf_func_warn
else:
    function = tf.function
