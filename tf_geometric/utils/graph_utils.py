# coding=utf-8
import tensorflow as tf
import numpy as np
import networkx as nx


def remove_self_loop_edge(edge_index, edge_weight=None):
    edge_index_is_tensor = tf.is_tensor(edge_index)
    edge_weight_is_tensor = edge_weight is not None and tf.is_tensor(edge_weight)

    if edge_index_is_tensor:
        edge_index = edge_index.numpy()

    if edge_weight_is_tensor:
        edge_weight = edge_weight.numpy()

    row, col = edge_index
    mask = row != col
    edge_index = edge_index[:, mask]
    if edge_weight is not None:
        edge_weight = edge_weight[mask]

    if edge_index_is_tensor:
        edge_index = tf.convert_to_tensor(edge_index)

    if edge_weight_is_tensor:
        edge_weight = tf.convert_to_tensor(edge_weight)

    return edge_index, edge_weight



# [[1,3,5], [2,1,4]] => [[1,3,5,2,1,4], [2,1,4,1,3,5]]
def convert_edge_to_directed(edge_index, edge_weight=None):

    edge_index_is_tensor = tf.is_tensor(edge_index)
    edge_weight_is_tensor = edge_weight is not None and tf.is_tensor(edge_weight)

    if edge_index_is_tensor:
        edge_index = edge_index.numpy()

    if edge_weight_is_tensor:
        edge_weight = edge_weight.numpy()

    g = nx.Graph()
    for i in range(edge_index.shape[1]):
        g.add_edge(edge_index[0, i], edge_index[1, i],
                   w=edge_weight[i] if edge_weight is not None else None)

    g = g.to_directed()

    edge_index = np.array(g.edges).T
    if edge_weight is not None:
        edge_weight = np.array([item[2] for item in g.edges.data("w")])

    if edge_index_is_tensor:
        edge_index = tf.convert_to_tensor(edge_index)

    if edge_weight_is_tensor:
        edge_weight = tf.convert_to_tensor(edge_weight)

    return edge_index, edge_weight


def add_self_loop_edge(edge_index, num_nodes, edge_weight=None, fill_weight=1.0):
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
