# coding=utf-8

import tensorflow as tf
from tf_geometric.nn.conv.gcn import gcn_cache_normed_edge

from tf_geometric.nn.conv.tagcn import tagcn


class TAGCN(tf.keras.Model):
    """
    The topology adaptive graph convolutional networks operator from the
     `"Topology Adaptive Graph Convolutional Networks"
     <https://arxiv.org/abs/1710.10370>`_ paper
    """

    def __init__(self, units, K=3, activation=None, use_bias=True,
                 renorm=False, improved=False,
                 kernel_regularizer=None, bias_regularizer=None,
                 *args, **kwargs):
        """

        :param units: Positive integer, dimensionality of the output space.
        :param K: Number of hops (default: '3").
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
        :param improved: Whether use improved GCN or not.

        """
        super().__init__(*args, **kwargs)
        self.units = units
        assert K > 0

        self.K = K

        self.activation = activation
        self.use_bias = use_bias

        self.kernel = None
        self.bias = None

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.renorm = renorm
        self.improved = improved

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[num_features * (self.K + 1), self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def cache_normed_edge(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        gcn_cache_normed_edge(graph, self.renorm, self.improved, override=override)

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

        return tagcn(x, edge_index, edge_weight, self.K, self.kernel,
                     bias=self.bias, activation=self.activation, renorm=self.renorm,
                     improved=self.improved, cache=cache)
