# coding=utf-8

import tensorflow as tf

from tf_geometric.nn.pool.sort_pool import sort_pool


class SortPool(tf.keras.Model):
    """
    OOP API for SortPool "An End-to-End Deep Learning Architecture for Graph Classification"
    """

    def __init__(self, K=None, ratio=None, sort_index=-1,  *args, **kwargs):
        """
        SAGPool

        :param score_gnn: A GNN model to score nodes for the pooling, [x, edge_index, edge_weight] => node_score.
        :param K: Keep top K targets for each source
        :param ratio: Keep num_targets * ratio targets for each source
        :param sort_index: The sort_index_th index of the last axis will used for sort.
        """
        super().__init__(*args, **kwargs)
        self.K = K
        self.ratio = ratio
        self.sort_index = sort_index

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight, node_graph_index]
        :return: Pooled grpah: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
        """
        x, edge_index, edge_weight, node_graph_index = inputs

        return sort_pool(x, edge_index, edge_weight, node_graph_index,
                         K=self.K, ratio=self.ratio, sort_index=self.sort_index, training=training)
