# coding=utf-8
import tensorflow as tf
from tf_geometric.data.graph import BatchGraph
from tf_geometric.nn.pool.topk_pool import topk_pool


def sag_pool(x, edge_index, edge_weight, node_graph_index,
             score_gnn, k=None, ratio=None,
             score_activation=None, training=None, cache=None):
    """
    Functional API for SAGPool

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param node_graph_index: Tensor/NDArray, shape: [num_nodes], graph index for each node
    :param score_gnn: A GNN model to score nodes for the pooling, [x, edge_index, edge_weight] => node_score.
    :param k: Keep top k targets for each source
    :param ratio: Keep num_targets * ratio targets for each source
    :param score_activation: Activation to use for node_score before multiplying node_features with node_score
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
    :return: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
    """

    if cache is None:
        node_score = score_gnn([x, edge_index, edge_weight], training=training)
    else:
        node_score = score_gnn([x, edge_index, edge_weight], training=training, cache=cache)

    topk_node_index = topk_pool(node_graph_index, node_score, k=k, ratio=ratio)

    if score_activation is not None:
        node_score = score_activation(node_score)

    pooled_graph = BatchGraph(
        x=x * node_score,
        edge_index=edge_index,
        node_graph_index=node_graph_index,
        edge_graph_index=None,
        edge_weight=edge_weight
    ).sample_new_graph_by_node_index(topk_node_index)

    return pooled_graph.x, pooled_graph.edge_index, pooled_graph.edge_weight, pooled_graph.node_graph_index
