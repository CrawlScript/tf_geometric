# coding=utf-8


import tensorflow as tf

from tf_geometric.nn.pool.min_cut_pool import min_cut_pool


class MinCutPool(tf.keras.Model):
    """
    OOP API for MinCutPool: "Spectral Clustering with Graph Neural Networks for Graph Pooling"
    """

    def __init__(self, feature_gnn, assign_gnn, units, num_clusters, activation=None, use_bias=True,
                 gnn_use_normed_edge=True,
                 bias_regularizer=None, *args, **kwargs):
        """
        MinCutPool

        :param feature_gnn: A GNN model to learn pooled node features, [x, edge_index, edge_weight] => updated_x,
            where updated_x corresponds to high-order node features.
        :param assign_gnn: A GNN model to learn cluster assignment for the pooling, [x, edge_index, edge_weight] => updated_x,
            where updated_x corresponds to the cluster assignment matrix.
        :param units: Positive integer, dimensionality of the output space. It must be provided if you set use_bias=True.
        :param num_clusters: Number of clusters for pooling.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector. If true, the "units" parameter must be provided.
        :param gnn_use_normed_edge: Boolean. Whether to use normalized edge for feature_gnn and assign_gnn.
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

        self.gnn_use_normed_edge = gnn_use_normed_edge

    def call(self, inputs, cache=None, training=None, mask=None, return_loss_func=False, return_losses=False):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight, node_graph_index]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :param return_loss_func: Boolean. If True, return (outputs, loss_func), where loss_func is a callable function
            that returns a list of losses.
        :param return_losses: Boolean. If True, return (outputs, losses), where losses is a list of losses.
        :return: Pooled graph: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
        """

        if return_loss_func and return_losses:
            raise Exception("return_loss_func and return_losses cannot be set to True at the same time")

        x, edge_index, edge_weight, node_graph_index = inputs

        outputs, loss_func = min_cut_pool(x, edge_index, edge_weight, node_graph_index,
                                          self.feature_gnn, self.assign_gnn, self.num_clusters,
                                          bias=self.bias, activation=self.activation,
                                          gnn_use_normed_edge=self.gnn_use_normed_edge,
                                          training=training, cache=cache,
                                          return_loss_func=True)
        self.add_loss(loss_func)

        if return_loss_func:
            return outputs, loss_func
        elif return_losses:
            losses = loss_func()
            return outputs, losses
        else:
            return outputs

