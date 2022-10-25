# coding=utf-8

import warnings

warnings.simplefilter('always', DeprecationWarning)

import tensorflow as tf

if tf.__version__[0] == "1":
    tf.enable_eager_execution()

from . import nn, utils, data, datasets, layers
from .data.graph import Graph, BatchGraph, HeteroDictGraph
# from .sparse.sparse_adj import SparseAdj
