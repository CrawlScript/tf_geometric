# coding=utf-8
import tensorflow as tf
from tensorflow import keras
from tf_geometric.nn.pool.common_pool import mean_pool


class MeanPool(keras.Model):

    def call(self, inputs, training=None, mask=None):
        if len(inputs) == 2:
            x, node_graph_index = inputs
            num_graphs = None
        else:
            x, node_graph_index, num_graphs = inputs

        return mean_pool(x, node_graph_index, num_graphs)