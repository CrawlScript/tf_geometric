# coding=utf-8

import tensorflow as tf
from tf_geometric.nn.conv.chebynet import chebynet_norm_edge, chebynet
from tf_geometric.layers.kernel.map_reduce import MapReduceGNN


class ChebyNet(MapReduceGNN):
    """
    The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    """

    def __init__(self, units, K, lambda_max, activation=tf.nn.relu, use_bias=True, normalization_type='sym',
                 *args, **kwargs):
        """

        :param units: Positive integer, dimensionality of the output space.
        :param K: Chebyshev filter size (default: '3").
        :param lambda_max:
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param activation: Activation function to use.
        :param normalization_type: The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`)

        """
        super().__init__(*args, **kwargs)
        self.units = units

        assert K > 0
        assert lambda_max is not None
        assert normalization_type in [None, 'sym', 'rw'], 'Invalid normalization'

        self.K = K

        self.use_bias = use_bias

        self.kernel = None
        self.bias = None

        self.activation = activation
        self.normalization_type = normalization_type
        self.lambda_max = lambda_max

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[self.K, num_features, self.units], initializer="glorot_uniform")
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

        return chebynet(x, edge_index, edge_weight, self.K, self.lambda_max, self.kernel, self.bias, self.activation,
                       self.normalization_type)
