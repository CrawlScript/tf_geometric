# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_mapper, identity_updater, \
    sum_updater
from tf_geometric.utils.graph_utils import add_self_loop_edge


def gin_norm_edge(edge_index, num_nodes, edge_weight=None, improved=False, cache=None):
    cache_key = "gin_normed_edge"

    if cache is not None and cache_key in cache and cache[cache_key] is not None:
        return cache[cache_key]

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    fill_weight = 2.0 if improved else 1.0
    edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight=edge_weight,
                                                 fill_weight=fill_weight)

    row, col = edge_index
    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=num_nodes)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(tf.math.is_inf(deg_inv_sqrt), tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
    deg_inv_sqrt = tf.where(tf.math.is_nan(deg_inv_sqrt), tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)

    noremd_edge_weight = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

    if cache is not None:
        cache[cache_key] = edge_index, noremd_edge_weight

    return edge_index, noremd_edge_weight


def gin_updater(x, reduced_neighbor_msg, eps):
    return x * (1 + eps) + reduced_neighbor_msg


def gin(x, edge_index, edge_weight, kernel, eps=0, bias=None, activation=None, improved=False, cache=None):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param improved: Whether use improved GIN or not.
    :param cache: A dict for caching A' for GIN. Different graph should not share the same cache dict.
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    updated_edge_index, normed_edge_weight = gin_norm_edge(edge_index, x.shape[0], edge_weight,
                                                           improved, cache)
    x = x @ kernel
    h = aggregate_neighbors(
        x, updated_edge_index, normed_edge_weight,
        identity_mapper,
        sum_reducer,
        identity_updater
    )

    h = gin_updater(x, h, eps)

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h
