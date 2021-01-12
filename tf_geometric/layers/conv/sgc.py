# coding=utf-8
from tf_geometric.nn.conv.sgc import sgc
from tf_geometric.nn.conv.gcn import gcn_cache_normed_edge
import tensorflow as tf


class SGC(tf.keras.Model):
    """
    The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper
    """

    def __init__(self, units, k=1, activation=None, use_bias=True, renorm=True, improved=False,
                 kernel_regularizer=None, bias_regularizer=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        :param units: Size of each output sample..
        :param k: Number of hops.(default: :obj:`1`)
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
        :param improved: Whether use improved GCN or not.
        """

        self.units = units
        self.use_bias = use_bias
        self.renorm = renorm
        self.improved = improved
        self.K = k
        self.activation = activation
        self.kernel = None
        self.bias = None

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shape):

        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
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
        :return: Updated node features (x), shape: [num_nodes, num_units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return sgc(x, edge_index, edge_weight, self.K, self.kernel,
                   bias=self.bias, activation=self.activation,
                   renorm=self.renorm, improved=self.improved, cache=cache)
