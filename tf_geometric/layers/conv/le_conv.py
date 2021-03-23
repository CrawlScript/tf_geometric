# coding=utf-8

import tensorflow as tf
from tf_geometric.nn.conv.le_conv import le_conv


class LEConv(tf.keras.Model):
    """
    Graph Convolutional Layer
    """

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, self.units],
                                           initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.self_use_bias:
            self.self_bias = self.add_weight("self_bias", shape=[self.units],
                                             initializer="zeros", regularizer=self.bias_regularizer)

        self.aggr_self_kernel = self.add_weight("aggr_self_kernel", shape=[num_features, self.units],
                                                initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.aggr_self_use_bias:
            self.aggr_self_bias = self.add_weight("aggr_self_bias", shape=[self.units],
                                                  initializer="zeros", regularizer=self.bias_regularizer)

        self.aggr_neighbor_kernel = self.add_weight("aggr_neighbor_kernel", shape=[num_features, self.units],
                                                    initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.aggr_neighbor_use_bias:
            self.aggr_neighbor_bias = self.add_weight("aggr_neighbor_bias", shape=[self.units],
                                                      initializer="zeros", regularizer=self.bias_regularizer)

    def __init__(self, units, activation=None,
                 self_use_bias=True,
                 aggr_self_use_bias=True,
                 aggr_neighbor_use_bias=False,
                 kernel_regularizer=None, bias_regularizer=None, *args, **kwargs):
        """

        :param units: Positive integer, dimensionality of the output space.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
        :param improved: Whether use improved GCN or not.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        """
        super().__init__(*args, **kwargs)
        self.units = units

        self.activation = activation

        self.self_use_bias = self_use_bias
        self.aggr_self_use_bias = aggr_self_use_bias
        self.aggr_neighbor_use_bias = aggr_neighbor_use_bias

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.self_kernel = None
        self.self_bias = None
        self.aggr_self_kernel = None
        self.aggr_self_bias = None
        self.aggr_neighbor_kernel = None
        self.aggr_neighbor_bias = None

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return le_conv(x, edge_index, edge_weight,
                       self.self_kernel, self.self_bias,
                       self.aggr_self_kernel, self.aggr_self_bias,
                       self.aggr_neighbor_kernel, self.aggr_neighbor_bias,
                       activation=self.activation)
