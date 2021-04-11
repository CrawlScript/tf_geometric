# coding=utf-8
import tensorflow as tf
from tf_geometric.data.graph import BatchGraph
from tf_geometric.nn.pool.topk_pool import topk_pool


def sort_pool(x, edge_index, edge_weight, node_graph_index,
              k=None, ratio=None,
              sort_index=-1, training=None):
    """
    Functional API for SortPool "An End-to-End Deep Learning Architecture for Graph Classification"

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param node_graph_index: Tensor/NDArray, shape: [num_nodes], graph index for each node
    :param k: Keep top k targets for each source
    :param ratio: Keep num_targets * ratio targets for each source
    :param sort_index: The sort_index_th index of the last axis will used for sort.
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
    """

    score = x[:, sort_index]
    topk_node_index = topk_pool(node_graph_index, score, k=k, ratio=ratio)

    pooled_graph = BatchGraph(
        x=x,
        edge_index=edge_index,
        node_graph_index=node_graph_index,
        edge_graph_index=None,
        edge_weight=edge_weight
    ).sample_new_graph_by_node_index(topk_node_index)

    return pooled_graph.x, pooled_graph.edge_index, pooled_graph.edge_weight, pooled_graph.node_graph_index