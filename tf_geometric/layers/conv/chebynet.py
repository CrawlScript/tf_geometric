# coding=utf-8

import warnings
import tensorflow as tf
from tf_geometric.nn.conv.chebynet import chebynet, chebynet_cache_normed_edge
from tf_geometric.nn.conv.gcn import gcn_build_cache_by_adj


class ChebyNet(tf.keras.Model):
    """
    The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    """

    def __init__(self, units, k, activation=None, use_bias=True, normalization_type="sym",
                 use_dynamic_lambda_max=False,
                 kernel_regularizer=None, bias_regularizer=None,
                 *args, **kwargs):
        """

        :param units: Positive integer, dimensionality of the output space.
        :param k: Chebyshev filter size (default: '3").
        :param lambda_max:
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param activation: Activation function to use.
        :param normalization_type: The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`)
        :param use_dynamic_lambda_max: If true, compute max eigen value for each forward,
            otherwise use 2.0 as the max eigen value
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        """
        super().__init__(*args, **kwargs)
        self.units = units

        assert k > 0
        assert normalization_type in [None, 'sym', 'rw'], 'Invalid normalization'

        self.k = k

        self.use_bias = use_bias

        self.kernels = []
        self.bias = None

        self.activation = activation
        self.normalization_type = normalization_type
        self.use_dynamic_lambda_max = use_dynamic_lambda_max

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        for k in range(self.k):
            kernel = self.add_weight("kernel{}".format(k), shape=[num_features, self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)
            self.kernels.append(kernel)

        # self.kernel = self.add_weight("kernel", shape=[self.K, num_features, self.units],
        #                               initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        chebynet_cache_normed_edge(graph, self.normalization_type,
                                   use_dynamic_lambda_max=self.use_dynamic_lambda_max, override=override)

    def cache_normed_edge(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None

        .. deprecated:: 0.0.56
            Use ``build_cache_for_graph`` instead.
        """
        warnings.warn(
            "'ChebyNet.cache_normed_edge(graph, override)' is deprecated, use 'ChebyNet.build_cache_for_graph(graph, override)' instead",
            DeprecationWarning)
        return self.build_cache_for_graph(graph, override=override)

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

        return chebynet(x, edge_index, edge_weight, self.k, self.kernels, self.bias, self.activation,
                        self.normalization_type, use_dynamic_lambda_max=self.use_dynamic_lambda_max, cache=cache)
