# coding=utf-8


import tensorflow as tf

from tf_geometric.nn.pool.diff_pool import diff_pool


class DiffPool(tf.keras.Model):
    """
    OOP API for DiffPool: "Hierarchical graph representation learning with differentiable pooling"
    """

    def __init__(self, feature_gnn, assign_gnn, units, num_clusters, activation=None, use_bias=True,
                 bias_regularizer=None, *args, **kwargs):
        """
        DiffPool

        :param feature_gnn: A GNN model to learn pooled node features, [x, edge_index, edge_weight] => updated_x,
            where updated_x corresponds to high-order node features.
        :param assign_gnn: A GNN model to learn cluster assignment for the pooling, [x, edge_index, edge_weight] => updated_x,
            where updated_x corresponds to the cluster assignment matrix.
        :param units: Positive integer, dimensionality of the output space. It must be provided if you set use_bias=True.
        :param num_clusters: Number of clusters for pooling.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector. If true, the "units" parameter must be provided.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        """
        super().__init__(*args, **kwargs)
        self.feature_gnn = feature_gnn
        self.assign_gnn = assign_gnn

        self.num_clusters = num_clusters
        self.activation = activation

        if use_bias and units is None:
            raise Exception("The \"units\" parameter is required when you set use_bias=True.")

        if use_bias:
            self.bias = self.add_weight("bias", shape=[units],
                                        initializer="zeros", regularizer=bias_regularizer)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight, node_graph_index]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Pooled graph: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
        """
        x, edge_index, edge_weight, node_graph_index = inputs

        return diff_pool(x, edge_index, edge_weight, node_graph_index,
                         self.feature_gnn, self.assign_gnn, self.num_clusters,
                         bias=self.bias, activation=self.activation, training=training, cache=cache)
