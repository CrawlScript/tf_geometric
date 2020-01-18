# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_updater, sum_reducer


def gcn_norm(edge_index, num_nodes, edge_weight=None):
    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    row, col = edge_index
    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=num_nodes)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(tf.math.is_inf(deg_inv_sqrt), tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
    deg_inv_sqrt = tf.where(tf.math.is_nan(deg_inv_sqrt), tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)

    return tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)


def gcn_mapper(repeated_x, neighbor_x, edge_weight=None):
    return neighbor_x * tf.expand_dims(edge_weight, 1)


def gcn(x, edge_index, normed_edge_weight, dense_w, dense_b=None, activation=None):
    x = x @ dense_w
    h = aggregate_neighbors(
        x, edge_index, normed_edge_weight,
        gcn_mapper,
        sum_reducer,
        sum_updater
    )

    if dense_b is not None:
        h += dense_b

    if activation is not None:
        h = activation(h)

    return h


def norm_and_gcn(x, edge_index, num_nodes, dense_w, edge_weight=None, dense_b=None, activation=None, return_norm=True):
    normed_edge_weight = gcn_norm(edge_index, num_nodes, edge_weight)
    outputs = gcn(x, edge_index, normed_edge_weight, dense_w, dense_b, activation)
    if return_norm:
        return outputs, normed_edge_weight
    else:
        return outputs






