# coding=utf-8

import tensorflow as tf
from tf_geometric.nn.pool.set2set import set2set


class Set2Set(tf.keras.Model):
    """
    OOP API for Set2Set
    """

    def __init__(self, num_iterations=4, *args, **kwargs):
        """
        Set2Set

        :param num_iterations: Number of iterations for attention.
        """
        super().__init__(*args, **kwargs)
        self.num_iterations = num_iterations
        self.lstm = None

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.lstm = tf.keras.layers.LSTM(num_features, return_sequences=True, return_state=True)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, node_graph_index]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Graph features, shape: [num_graphs, num_node_features * 2]
        """
        x, node_graph_index = inputs

        return set2set(x, node_graph_index, self.lstm, self.num_iterations, training=training)
