# coding=utf-8

from tf_geometric.nn.conv.gcn import gcn_build_cache_for_graph, gcn_build_cache_by_adj
from tf_geometric.nn.conv.ssgc import ssgc
import tensorflow as tf
import warnings


class SSGC(tf.keras.Model):

    def build(self, input_shapes):

        if self.units_list is None:
            return

        x_shape = input_shapes[0]
        last_units = x_shape[-1]

        for i, units in enumerate(self.units_list):
            kernel_name = "kernel_{}".format(i)
            kernel = self.add_weight(kernel_name, shape=[last_units, units],
                                     initializer="glorot_uniform", regularizer=self.kernel_regularizer)

            bias_name = "bias_{}".format(i)
            bias = self.add_weight(bias_name, shape=[units],
                                   initializer="zeros", regularizer=self.bias_regularizer)
            last_units = units

            self.kernels.append(kernel)
            self.biases.append(bias)

            # setattr(self, kernel_name, kernel)
            # setattr(self, bias_name, bias)

    def __init__(self,
                 units_list=None, k=10, alpha=0.1,
                 dense_activation=tf.nn.relu, activation=None,
                 dense_drop_rate=0.0,
                 last_dense_drop_rate=0.0,
                 edge_drop_rate=0.0,
                 kernel_regularizer=None, bias_regularizer=None, *args, **kwargs):
        """

        OOP API for Simple Spectral Graph Convolution (SSGC / S^2GC).
        Paper URL: https://openreview.net/forum?id=CYO5T-YjWZV

        :param units_list: List of Positive integers consisting of dimensionality of the output space of each dense layer.
        :param k: Number of propagation power iterations.
        :param dense_activation: Activation function to use for the dense layers,
            except for the last dense layer, which will not be activated.
        :param activation: Activation function to use for the output.
        :param alpha: Teleport Probability.
        :param dense_drop_rate: Dropout rate for the input of every dense layer.
        :param last_dense_drop_rate: Dropout rate for the output of the last dense layer.
            last_dense_drop_rate is usually set to 0.0 for classification tasks.
        :param edge_drop_rate: Dropout rate for the edges/adj used for propagation.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrices.
        :param bias_regularizer: Regularizer function applied to the bias vectors.
        """

        super().__init__(*args, **kwargs)
        self.k = k
        self.alpha = alpha
        self.activation = activation

        self.units_list = units_list
        self.dense_activation = dense_activation

        self.dense_drop_rate = dense_drop_rate
        self.last_dense_drop_rate = last_dense_drop_rate
        self.edge_drop_rate = edge_drop_rate

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.kernels = []
        self.biases = []


    def build_cache_by_adj(self, sparse_adj, override=False, cache=None):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        return gcn_build_cache_by_adj(sparse_adj, override=override, cache=cache)

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        gcn_build_cache_for_graph(graph, override=override)

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
            "'SSGC.cache_normed_edge(graph, override)' is deprecated, use 'SSGC.build_cache_for_graph(graph, override)' instead",
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

        return ssgc(x=x, edge_index=edge_index, edge_weight=edge_weight,
                    kernels=self.kernels, biases=self.biases,
                    k=self.k, alpha=self.alpha,
                    dense_activation=self.dense_activation, activation=self.activation,
                    dense_drop_rate=self.dense_drop_rate,
                    last_dense_drop_rate=self.last_dense_drop_rate,
                    edge_drop_rate=self.edge_drop_rate,
                    cache=cache, training=training)
