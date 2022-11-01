# coding=utf-8
import tf_sparse as tfs

from tf_geometric.nn.conv.gcn import gcn, gcn_build_cache_for_graph, gcn_build_cache_by_adj
import tensorflow as tf
import warnings
import numpy as np

from tf_geometric.utils.tf_sparse_utils import compute_num_or_size_splits


class GCN(tf.keras.Model):
    """
    Graph Convolutional Layer
    """

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        if self.num_splits is not None:
            num_h_features = self.units if self.use_kernel else num_features
            self.num_or_size_splits = compute_num_or_size_splits(num_h_features, self.num_splits)

        if self.use_kernel:
            self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
                                          initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units if self.use_kernel else num_features],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def __init__(self, units, activation=None,
                 use_kernel=True,
                 use_bias=True,
                 norm="both", add_self_loop=True, sym=True,
                 renorm=True, improved=False,
                 edge_drop_rate=0.0,
                 num_splits=None,
                 num_or_size_splits=None,
                 kernel_regularizer=None, bias_regularizer=None, *args, **kwargs):
        """

        :param units: Positive integer, dimensionality of the output space.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param norm: normalization mode both|left|right:
            - both: (D^(-1/2)A)D^(-1/2);
            - left: D^(-1/2)A; 
            - right: AD^(-1/2);
        :param add_self_loop: Whether add self-loop to adj during normalization.
        :param sym: Optional, only used when norm=="both". Setting sym=True indicates that the input
            sparse_adj is symmetric.
        :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
        :param improved: Whether use improved GCN or not.
        :param edge_drop_rate: Dropout rate of the propagation weights.
        :param num_or_size_splits: Split (XW) to compute A(XW) for large graphs (Not affecting the output).
            See the num_or_size_splits param of the tf.split API.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        """
        super().__init__(*args, **kwargs)
        self.units = units

        self.activation = activation
        self.use_kernel = use_kernel
        self.use_bias = use_bias

        self.edge_drop_rate = edge_drop_rate

        self.kernel = None
        self.bias = None

        self.norm = norm
        self.add_self_loop = add_self_loop
        self.sym = sym
        self.renorm = renorm
        self.improved = improved

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        if num_splits is not None and num_or_size_splits is not None:
            raise Exception("cannot provide both num_splits and num_or_size_splits for GCN")

        self.num_splits = num_splits
        self.num_or_size_splits = num_or_size_splits

    def build_cache_by_adj(self, sparse_adj, override=False, cache=None):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        return gcn_build_cache_by_adj(sparse_adj,
                                      self.norm, self.add_self_loop, self.sym,
                                      self.renorm, self.improved, override=override, cache=cache)

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        gcn_build_cache_for_graph(graph,
                                  self.norm, self.add_self_loop, self.sym,
                                  self.renorm, self.improved, override=override)

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
        warnings.warn("'GCN.cache_normed_edge(graph, override)' is deprecated, use 'GCN.build_cache_for_graph(graph, override)' instead", DeprecationWarning)
        return self.build_cache_for_graph(graph, override=override)

    def call(self, inputs, cache=None, split=True, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, sparse_adj], [x, edge_index], 
            or [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :param split: bool, whether split (XW) to compute A(XW) if self.num_splits is not None
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if isinstance(inputs[1], tfs.SparseMatrix):
            x, sparse_adj = inputs
        elif len(inputs) == 3:
            x, edge_index, edge_weight = inputs
            num_nodes = tfs.shape(x)[0]
            sparse_adj = tfs.SparseMatrix(edge_index, value=edge_weight, shape=[num_nodes, num_nodes])
        elif len(inputs) == 2:
            x, edge_index = inputs
            num_nodes = tfs.shape(x)[0]
            sparse_adj = tfs.SparseMatrix(edge_index, shape=[num_nodes, num_nodes])

        return gcn(x, sparse_adj, self.kernel, self.bias,
                   activation=self.activation,
                   norm=self.norm, add_self_loop=self.add_self_loop, sym=self.sym,
                   renorm=self.renorm, improved=self.improved,
                   edge_drop_rate=self.edge_drop_rate,
                   num_or_size_splits=self.num_or_size_splits if split else None,
                   training=training, cache=cache)
