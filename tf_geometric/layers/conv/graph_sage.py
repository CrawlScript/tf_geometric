import tensorflow as tf
from tensorflow import keras
from tf_geometric.nn.conv.graph_sage import mean_graph_sage, gcn_graph_sage, mean_pool_graph_sage, \
    max_pool_graph_sage, lstm_graph_sage
from tf_geometric.layers.kernel.map_reduce import MapReduceGNN


"""
The GraphSAGE operator from the `"Inductive Representation Learning on
Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
"""
class MeanGraphSage(MapReduceGNN):

    def __init__(self, units, activation=tf.nn.relu, use_bias=True,
                 normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.normalize = normalize

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.neighs_kernel = self.add_weight("neighs_kernel", shape=[num_features, self.units],
                                             initializer="glorot_uniform")
        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, self.units],
                                           initializer="glorot_uniform")

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")

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

        return mean_graph_sage(x, edge_index, edge_weight, self.neighs_kernel, self.self_kernel, self.bias,
                               self.activation, self.normalize)


class GCNGraphSage(MapReduceGNN):
    def __init__(self, units, activation=tf.nn.relu, use_bias=True,
                 normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.normalize = normalize

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform")

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

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

        return gcn_graph_sage(x, edge_index, edge_weight, self.kernel, self.bias, self.activation, self.normalize, cache=cache)


class MeanPoolGraphSage(MapReduceGNN):
    def __init__(self, units, activation=tf.nn.relu, use_bias=True,
                 normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.normalize = normalize

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.mlp_kernel = self.add_weight("mlp_kernel", shape=[num_features, self.units*4],
                                          initializer="glorot_uniform")
        if self.use_bias:
            self.mlp_bias = self.add_weight("mlp_bias", shape=[self.units*4], initializer="zeros")
        # self.mlp_kernel = keras.layers.Dense(self.units, input_dim=2, use_bias=True, kernel_regularizer= tf.nn.l2_normalize, activation=tf.nn.relu)

        self.neighs_kernel = self.add_weight("neighs_kernel", shape=[self.units*4, self.units],
                                             initializer="glorot_uniform")
        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, self.units],
                                           initializer="glorot_uniform")

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")


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

        return mean_pool_graph_sage(x, edge_index, edge_weight, self.mlp_kernel, self.neighs_kernel,
                                    self.self_kernel, self.mlp_bias, self.bias, self.activation,
                                    self.normalize)


class MaxPoolGraphSage(MapReduceGNN):
    def __init__(self, units, activation=tf.nn.relu, use_bias=True,
                 normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.normalize = normalize

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.mlp_kernel = self.add_weight("mlp_kernel", shape=[num_features, self.units*4],
                                          initializer="glorot_uniform")
        if self.use_bias:
            self.mlp_bias = self.add_weight("mlp_bias", shape=[self.units*4], initializer="zeros")
        self.neighs_kernel = self.add_weight("neighs_kernel", shape=[self.units*4, self.units],
                                             initializer="glorot_uniform")
        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, self.units],
                                           initializer="glorot_uniform")

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")

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

        return max_pool_graph_sage(x, edge_index, edge_weight, self.mlp_kernel, self.neighs_kernel,
                                   self.self_kernel,
                                   self.mlp_bias, self.bias, self.activation, self.normalize)


class LSTMGraphSage(MapReduceGNN):

    def __init__(self, units, activation=tf.nn.relu, use_bias=True,
                 normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.normalize = normalize

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.lstm = tf.keras.layers.LSTM(self.units, kernel_regularizer=tf.nn.l2_normalize, return_sequences=True)
        self.neighs_kernel = self.add_weight("neighs_kernel", shape=[self.units, self.units],
                                             initializer="glorot_uniform")
        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, self.units],
                                           initializer="glorot_uniform")

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")


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

        return lstm_graph_sage(x, edge_index, edge_weight, self.lstm, self.neighs_kernel,
                                       self.self_kernel, self.bias, self.activation,
                                      self.normalize)

