# coding=utf-8

import tensorflow as tf
from tf_geometric.nn.conv.graph_sage import mean_graph_sage, gcn_graph_sage, mean_pool_graph_sage, \
    max_pool_graph_sage, lstm_graph_sage, sum_graph_sage


class MeanGraphSage(tf.keras.Model):
    """
    GraphSAGE: `"Inductive Representation Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    """

    def __init__(self, units, activation=tf.nn.relu, use_bias=True, concat=True,
                 normalize=False,
                 kernel_regularizer=None, bias_regularizer=None,
                 *args, **kwargs):
        """

        :param units:
        :param activation:
        :param use_bias:
        :param concat:
        :param normalize:
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.concat = concat
        self.normalize = normalize

        if concat and (units % 2 != 0):
            raise Exception("units must be a event number if concat is True")

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.self_kernel = None
        self.neighbor_kernel = None
        self.bias = None

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        if self.concat:
            kernel_units = self.units // 2
        else:
            kernel_units = self.units

        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, kernel_units],
                                           initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.neighbor_kernel = self.add_weight("neighbor_kernel", shape=[num_features, kernel_units],
                                               initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """
        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return mean_graph_sage(x, edge_index, edge_weight,
                               self.self_kernel,
                               self.neighbor_kernel,
                               bias=self.bias,
                               activation=self.activation, concat=self.concat,
                               normalize=self.normalize)

class SumGraphSage(tf.keras.Model):
    """
    GraphSAGE: `"Inductive Representation Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    """

    def __init__(self, units, activation=tf.nn.relu, use_bias=True, concat=True,
                 normalize=False,
                 kernel_regularizer=None, bias_regularizer=None,
                 *args, **kwargs):
        """

        :param units:
        :param activation:
        :param use_bias:
        :param concat:
        :param normalize:
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.concat = concat
        self.normalize = normalize

        if concat and (units % 2 != 0):
            raise Exception("units must be a event number if concat is True")

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.self_kernel = None
        self.neighbor_kernel = None
        self.bias = None

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        if self.concat:
            kernel_units = self.units // 2
        else:
            kernel_units = self.units

        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, kernel_units],
                                           initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.neighbor_kernel = self.add_weight("neighbor_kernel", shape=[num_features, kernel_units],
                                               initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """
        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return sum_graph_sage(x, edge_index, edge_weight,
                               self.self_kernel,
                               self.neighbor_kernel,
                               bias=self.bias,
                               activation=self.activation, concat=self.concat,
                               normalize=self.normalize)


class GCNGraphSage(tf.keras.Model):
    def __init__(self, units, activation=tf.nn.relu, use_bias=True,
                 normalize=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.normalize = normalize

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return gcn_graph_sage(x, edge_index, edge_weight, self.kernel, self.bias, self.activation, self.normalize,
                              cache=cache)


class MeanPoolGraphSage(tf.keras.Model):
    def __init__(self, units, activation=tf.nn.relu, use_bias=True, concat=True,
                 normalize=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.concat = concat

        if concat and (units % 2 != 0):
            raise Exception("units must be a event number if concat is True")

        self.normalize = normalize

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.self_kernel = None
        self.neighbor_mlp_kernel = None
        self.neighbor_mlp_bias = None
        self.neighbor_kernel = None


        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        if self.concat:
            kernel_units = self.units // 2
        else:
            kernel_units = self.units

        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, kernel_units],
                                           initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        self.neighbor_mlp_kernel = self.add_weight("neighbor_mlp_kernel", shape=[num_features, kernel_units * 4],
                                                   initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.neighbor_mlp_bias = self.add_weight("neighbor_mlp_bias", shape=[kernel_units * 4],
                                                     initializer="zeros", regularizer=self.bias_regularizer)

        self.neighbor_kernel = self.add_weight("neighbor_kernel", shape=[kernel_units * 4, kernel_units],
                                               initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return mean_pool_graph_sage(x, edge_index, edge_weight,
                                    self.self_kernel,
                                    self.neighbor_mlp_kernel,
                                    self.neighbor_kernel,
                                    neighbor_mlp_bias=self.neighbor_mlp_bias,
                                    bias=self.bias, activation=self.activation,
                                    concat=self.concat, normalize=self.normalize)


class MaxPoolGraphSage(tf.keras.Model):
    def __init__(self, units, activation=tf.nn.relu, use_bias=True,
                 concat=True, normalize=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.concat = concat

        if concat and (units % 2 != 0):
            raise Exception("units must be a event number if concat is True")

        self.normalize = normalize

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.self_kernel = None
        self.neighbor_mlp_kernel = None
        self.neighbor_mlp_bias = None
        self.neighbor_kernel = None
        self.bias = None

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        if self.concat:
            kernel_units = self.units // 2
        else:
            kernel_units = self.units

        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, kernel_units],
                                           initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        self.neighbor_mlp_kernel = self.add_weight("mlp_kernel", shape=[num_features, kernel_units * 4],
                                                   initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.neighbor_mlp_bias = self.add_weight("mlp_bias", shape=[kernel_units * 4],
                                                     initializer="zeros", regularizer=self.bias_regularizer)
        self.neighbor_kernel = self.add_weight("neighs_kernel", shape=[kernel_units * 4, kernel_units],
                                               initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return max_pool_graph_sage(x, edge_index, edge_weight,
                                   self.self_kernel,
                                   self.neighbor_mlp_kernel,
                                   self.neighbor_kernel,
                                   neighbor_mlp_bias=self.neighbor_mlp_bias,
                                   bias=self.bias, activation=self.activation,
                                   concat=self.concat, normalize=self.normalize)


class LSTMGraphSage(tf.keras.Model):

    def __init__(self, units, activation=tf.nn.relu, use_bias=True,
                 concat=True, normalize=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.concat = concat
        self.normalize = normalize

        if concat and (units % 2 != 0):
            raise Exception("units must be a event number if concat is True")

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.lstm = None
        self.self_kernel = None
        self.neighbor_kernel = None
        self.bias = None

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        if self.concat:
            kernel_units = self.units // 2
        else:
            kernel_units = self.units

        self.lstm = tf.keras.layers.LSTM(kernel_units, return_sequences=True,
                                         kernel_regularizer=self.kernel_regularizer,
                                         bias_regularizer=self.bias_regularizer)

        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, kernel_units],
                                           initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        self.neighbor_kernel = self.add_weight("neighbor_kernel", shape=[kernel_units, kernel_units],
                                               initializer="glorot_uniform", regularizer=self.kernel_regularizer)


        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index] or [x, edge_index, edge_weight].
            Note that the edge_weight will not be used.
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        x, edge_index = inputs[0], inputs[1]

        return lstm_graph_sage(x, edge_index, self.lstm,
                               self.self_kernel,
                               self.neighbor_kernel,
                               bias=self.bias, activation=self.activation,
                               concat=self.concat, normalize=self.normalize, training=training)
