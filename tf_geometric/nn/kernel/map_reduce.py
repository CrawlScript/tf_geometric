# coding=utf-8

import tensorflow as tf
from tf_geometric.nn.kernel.segment import segment_op_with_pad


def identity_mapper(repeated_x, neighbor_x, edge_weight=None):
    return neighbor_x


def neighbor_count_mapper(repeated_x, neighbor_x, edge_weight=None):
    return tf.ones([neighbor_x.shape[0], 1])


def sum_reducer(neighbor_msg, node_index, num_nodes=None):
    return tf.math.unsorted_segment_sum(neighbor_msg, node_index, num_segments=num_nodes)


def sum_updater(x, reduced_neighbor_msg):
    return x + reduced_neighbor_msg


def identity_updater(x, reduced_neighbor_msg):
    return reduced_neighbor_msg


def mean_reducer(neighbor_msg, node_index, num_nodes=None):
    return tf.math.unsorted_segment_mean(neighbor_msg, node_index, num_segments=num_nodes)


def max_reducer(neighbor_msg, node_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = tf.reduce_max(node_index) + 1
    # return tf.math.unsorted_segment_max(neighbor_msg, node_index, num_segments=num_nodes)
    # max_x = tf.math.unsorted_segment_max(x, node_graph_index, num_segments=num_graphs)
    max_neighbor_msg = segment_op_with_pad(tf.math.segment_max, neighbor_msg, node_index, num_segments=num_nodes)
    return max_neighbor_msg


def aggregate_neighbors(x, edge_index, edge_weight=None, mapper=identity_mapper, reducer=sum_reducer, updater=sum_updater):
    """
    :param x:
    :param edge_index:
    :param mapper: (features_of_node, features_of_neighbor_node, edge_weight) => neighbor_msg
    :param reducer: (neighbor_msg, node_index) => reduced_neighbor_msg
    :param updater: (features_of_node, reduced_neighbor_msg, num_nodes) => aggregated_node_features
    :return:
    """

    if edge_index.shape[0] == 0:
        return x

    row, col = edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_msg = mapper(repeated_x, neighbor_x, edge_weight=edge_weight)
    reduced_msg = reducer(neighbor_msg, row, num_nodes=x.shape[0])

    udpated = updater(x, reduced_msg)
    return udpated






