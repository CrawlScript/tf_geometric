# coding=utf-8
import tensorflow as tf
from tensorflow import keras
from tf_geometric.nn.pooling.mean_pooling import mean_pooling


class MeanPooling(keras.Model):

    def call(self, inputs, training=None, mask=None):
        if len(inputs) == 2:
            x, node_graph_index = inputs
            num_graphs = None
        else:
            x, node_graph_index, num_graphs = inputs

        return mean_pooling(x, node_graph_index, num_graphs)