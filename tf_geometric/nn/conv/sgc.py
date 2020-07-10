#coding=utf-8
import tensorflow as tf

from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_updater, sum_reducer, identity_updater
from tf_geometric.utils.graph_utils import add_self_loop_edge


def mean_reducer(neighbor_msg, node_index, num_nodes=None):
    return tf.math.unsorted_segment_mean(neighbor_msg, node_index, num_segments=num_nodes)

def gcn_norm_edge(edge_index, num_nodes, edge_weight=None, renorm=True, improved=False, cache=None):
    cache_key = "gcn_normed_edge"

    if cache is not None and cache_key in cache and cache[cache_key] is not None:
        return cache[cache_key]

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    fill_weight = 2.0 if improved else 1.0

    if renorm:
        edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight=edge_weight, fill_weight=fill_weight)

    row, col = edge_index
    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=num_nodes)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    normed_edge_weight = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

    if not renorm:
        edge_index, normed_edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight=normed_edge_weight,
                                                            fill_weight=fill_weight)

    if cache is not None:
        cache[cache_key] = edge_index, normed_edge_weight

    return edge_index, normed_edge_weight

def gcn_mapper(repeated_x, neighbor_x, edge_weight=None):
    return neighbor_x * tf.expand_dims(edge_weight, 1)

def sgc(x, edge_index, edge_weight):
    """
    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :return: Updated node features (x), shape: [num_nodes, num_features]
    """

    h = aggregate_neighbors(
        x, edge_index, edge_weight,
        gcn_mapper,
        sum_reducer,
        identity_updater
    )


    return h