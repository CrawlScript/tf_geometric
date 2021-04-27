# coding=utf-8
import tensorflow as tf

if tf.__version__[0] == "1":
    tf.enable_eager_execution()


from . import nn, utils, data, datasets, layers
from tf_geometric.data.graph import Graph, BatchGraph



