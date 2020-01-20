# coding=utf-8
import tensorflow as tf

if tf.__version__[0] == "1":
    tf.enable_eager_execution()

import tf_geometric.nn as nn
import tf_geometric.utils as utils
import tf_geometric.data as data
import tf_geometric.datasets as datasets
import tf_geometric.layers as layers

from tf_geometric.data.graph import Graph, BatchGraph



