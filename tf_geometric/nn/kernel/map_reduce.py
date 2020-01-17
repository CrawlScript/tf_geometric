# coding=utf-8

import tensorflow as tf


def neighbor_count_mapper(repeated_x, neighbor_x):
    return tf.ones([neighbor_x.shape[0], 1])


def sum_reducer(neighbor_msg, node_index):
    return tf.segment_sum(neighbor_msg, node_index)


def sum_updater(x, reduced_neighbor_msg):
    return x + reduced_neighbor_msg


def aggregate_neighbors(x, edge_index, mapper, reducer=sum_reducer, updater=sum_updater, directed=False):
    """
    :param x:
    :param edge_index:
    :param mapper: (features_of_node, features_of_neighbor_node) => neighbor_msg
    :param reducer: (neighbor_msg, node_index) => reduced_neighbor_msg
    :param updater: (features_of_node, reduced_neighbor_msg) => aggregated_node_features
    :return:
    """

    if not directed:
        edge_index = tf.concat([edge_index, edge_index[[1, 0]]], axis=1)

    # from adjacency matrix view, row corresponds to node, col corresponds to neighbor
    row, col = edge_index
    repeated_x = x[row]
    neighbor_x = x[col]

    neighbor_msg = mapper(repeated_x, neighbor_x)
    reduced_msg = reducer(neighbor_msg, row)
    udpated = updater(x, reduced_msg)
    return udpated


