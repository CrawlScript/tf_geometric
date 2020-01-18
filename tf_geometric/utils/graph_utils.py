# coding=utf-8
import tensorflow as tf
import numpy as np


# [[1,3,5], [2,1,4]] => [[1,3,5,2,1,4], [2,1,4,1,3,5]]
def convert_edge_index_to_undirected(edge_index, edge_weight=None):
    row, col = edge_index
    upper_mask = row < col
    edge_index = edge_index[:, upper_mask]

    edge_index = np.concatenate([edge_index, edge_index[[1, 0]]], axis=1)

    if edge_weight is not None:
        edge_weight = edge_weight[upper_mask]
        edge_weight = np.concatenate([edge_weight, edge_weight], axis=0)

    return edge_index, edge_weight


def add_diagonal_edge_index(edge_index, num_nodes, edge_weight=None, fill_weight=1.0):
    diagnal_edges = [[node_index, node_index] for node_index in range(num_nodes)]
    diagnal_edge_index = np.array(diagnal_edges).T.astype(np.int32)

    updated_edge_index = tf.concat([edge_index, diagnal_edge_index], axis=1)

    if not tf.is_tensor(edge_index):
        updated_edge_index = updated_edge_index.numpy()

    if edge_weight is not None:
        diagnal_edge_weight = tf.cast(tf.fill([num_nodes], fill_weight), tf.float32)
        updated_edge_weight = tf.concat([edge_weight, diagnal_edge_weight], axis=0)

        if not tf.is_tensor(edge_weight):
            updated_edge_weight = updated_edge_weight.numpy()
    else:
        updated_edge_weight = None

    return updated_edge_index, updated_edge_weight
