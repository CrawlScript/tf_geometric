# coding=utf-8
import tensorflow as tf


def le_conv(x, edge_index, edge_weight,
            self_kernel, self_bias,
            aggr_self_kernel, aggr_self_bias,
            aggr_neighbor_kernel, aggr_neighbor_bias, activation=None):
    """
    Functional API for LeConv in ASAP.

    h_i = activation(x_i @ self_kernel + \sum_{j} (x_i @ aggr_self_kernel - x_j @ aggr_neighbor_kernel))

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param self_kernel: Please look at the formula above.
    :param aggr_self_kernel: Please look at the formula above.
    :param aggr_neighbor_kernel: Please look at the formula above.
    :param activation: Activation function to use.
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    if edge_weight is None:
        edge_weight = tf.ones([tf.shape(edge_index)[1]], dtype=tf.float32)

    num_nodes = tf.shape(x)[0]
    self_h = x @ self_kernel
    if self_bias is not None:
        self_h += self_bias

    aggr_self_h = x @ aggr_self_kernel
    if aggr_self_bias is not None:
        aggr_self_h += aggr_self_bias

    aggr_neighbor_h = x @ aggr_neighbor_kernel
    if aggr_neighbor_bias is not None:
        aggr_neighbor_h += aggr_neighbor_bias

    row, col = edge_index[0], edge_index[1]

    repeated_aggr_self_h = tf.gather(aggr_self_h, col)
    repeated_aggr_neighbor_h = tf.gather(aggr_neighbor_h, col)
    repeated_aggr_h = (repeated_aggr_self_h - repeated_aggr_neighbor_h) * tf.expand_dims(edge_weight, axis=-1)
    aggr_h = tf.math.unsorted_segment_sum(repeated_aggr_h, row, num_nodes)

    h = self_h + aggr_h

    if activation is not None:
        h = activation(h)

    return h



