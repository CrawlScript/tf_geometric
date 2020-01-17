# coding=utf-8

import tensorflow as tf

from tf_geometric.nn.kernel.segment import segment_sum_with_pad


def identity_mapper(repeated_x, neighbor_x, edge_weight=None):
    return neighbor_x


def neighbor_count_mapper(repeated_x, neighbor_x, edge_weight=None):
    return tf.ones([neighbor_x.shape[0], 1])


def sum_reducer(neighbor_msg, node_index, num_nodes=None):
    return segment_sum_with_pad(neighbor_msg, node_index, total=num_nodes)


def sum_updater(x, reduced_neighbor_msg):
    return x + reduced_neighbor_msg


def aggregate_neighbors(x, edge_index, edge_weight=None, mapper=identity_mapper, reducer=sum_reducer, updater=sum_updater, ):
    """
    :param x:
    :param edge_index:
    :param mapper: (features_of_node, features_of_neighbor_node, edge_weight) => neighbor_msg
    :param reducer: (neighbor_msg, node_index) => reduced_neighbor_msg
    :param updater: (features_of_node, reduced_neighbor_msg, num_nodes) => aggregated_node_features
    :return:
    """

    row, col = edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_msg = mapper(repeated_x, neighbor_x, edge_weight=edge_weight)
    reduced_msg = reducer(neighbor_msg, row, num_nodes=len(x))

    udpated = updater(x, reduced_msg)
    return udpated


