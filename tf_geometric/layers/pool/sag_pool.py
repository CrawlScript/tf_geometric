# coding=utf-8

import tensorflow as tf

from tf_geometric.nn.pool.sag_pool import sag_pool


class SAGPool(tf.keras.Model):
    """
    OOP API for SAGPool
    """

    def __init__(self, score_gnn, k=None, ratio=None, score_activation=None, *args, **kwargs):
        """
        SAGPool

        :param score_gnn: A GNN model to score nodes for the pooling, [x, edge_index, edge_weight] => node_score.
        :param k: Keep top k targets for each source
        :param ratio: Keep num_targets * ratio targets for each source
        :param score_activation: Activation to use for node_score before multiplying node_features with node_score
        """
        super().__init__(*args, **kwargs)
        self.score_gnn = score_gnn
        self.k = k
        self.ratio = ratio
        self.score_activation = score_activation

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight, node_graph_index]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Pooled graph: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
        """
        x, edge_index, edge_weight, node_graph_index = inputs

        return sag_pool(x, edge_index, edge_weight, node_graph_index, self.score_gnn,
                        k=self.k, ratio=self.ratio, score_activation=self.score_activation,
                        training=training, cache=cache)
