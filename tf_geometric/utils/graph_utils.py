# coding=utf-8
import tensorflow as tf
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

from tf_geometric.utils.union_utils import convert_union_to_numpy


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


def negative_sampling(num_samples, num_nodes, edge_index=None, replace=True):
    """

    :param num_samples:
    :param num_nodes:
    :param edge_index: if edge_index is provided, sampled positive edges will be filtered
    :param replace: only works when edge_index is provided, deciding whether sampled edges should be unique
    :return:
    """

    edge_index = convert_union_to_numpy(edge_index, np.int32)

    if edge_index is None:
        sampled_edge_index = np.random.randint(0, num_nodes, [2, num_samples]).astype(np.int32)
    else:
        # slow
        # edge_set = set([tuple(edge) for edge in edge_index.T])
        # sampled_edges = []
        # for _ in num_samples:
        #     while True:
        #         a, b = np.random.randint(0, num_nodes, [2], dtype=np.int32)
        #         if a == b:
        #             continue
        #         edge = (a, b)
        #         if edge not in edge_set:
        #             sampled_edges.append(edge)
        #             break
        # sampled_edge_index = np.array(sampled_edges, dtype=np.int32).T

        # fast
        adj = np.ones([num_nodes, num_nodes])
        np.fill_diagonal(adj, 0)
        adj[edge_index[0], edge_index[1]] = 0
        neg_edges = np.nonzero(adj)
        neg_edge_index = np.stack(neg_edges, axis=0)
        random_indices = np.random.choice(list(range(neg_edge_index.shape[1])), num_samples, replace=replace)
        sampled_edge_index = neg_edge_index[:, random_indices].astype(np.int32)

    if tf.is_tensor(edge_index):
        sampled_edge_index = tf.convert_to_tensor(sampled_edge_index)

    return sampled_edge_index


def negative_sampling_with_start_node(start_node_index, num_nodes, edge_index=None):
    """

    :param start_node_index: Tensor or ndarray
    :param num_nodes:
    :param edge_index: if edge_index is provided, sampled positive edges will be filtered
    :return:
    """

    start_node_index_is_tensor = tf.is_tensor(start_node_index)

    start_node_index = convert_union_to_numpy(start_node_index, dtype=np.int32)
    edge_index = convert_union_to_numpy(edge_index, np.int32)
    num_samples = len(start_node_index)

    if edge_index is None:
        end_node_index = np.random.randint(0, num_nodes, [num_samples]).astype(np.int32)
        sampled_edge_index = np.stack([start_node_index, end_node_index], axis=0)
    else:
        edge_set = set([tuple(edge) for edge in edge_index.T])

        sampled_edges = []
        for a in start_node_index:
            while True:
                b = np.random.randint(0, num_nodes, dtype=np.int32)
                if a == b:
                    continue
                edge = (a, b)
                if edge not in edge_set:
                    sampled_edges.append(edge)
                    break

        sampled_edge_index = np.array(sampled_edges, dtype=np.int32).T

    if start_node_index_is_tensor:
        sampled_edge_index = tf.convert_to_tensor(sampled_edge_index)

    return sampled_edge_index


def extract_unique_edge(edge_index, edge_weight=None, mode="undirected"):
    is_edge_index_tensor = tf.is_tensor(edge_index)
    is_edge_weight_tensor = tf.is_tensor(edge_weight)

    edge_index = convert_union_to_numpy(edge_index, dtype=np.int32)
    edge_weight = convert_union_to_numpy(edge_weight, dtype=np.float32)

    edge_set = set()
    unique_edge_index = []
    for i in range(edge_index.shape[1]):
        edge = edge_index[:, i]
        if mode == "undirected":
            edge = sorted(edge)
        edge = tuple(edge)

        if edge in edge_set:
            continue
        else:
            unique_edge_index.append(i)
            edge_set.add(edge)

    edge_index = edge_index[:, unique_edge_index]
    if is_edge_index_tensor:
        edge_index = tf.convert_to_tensor(edge_index)

    if edge_weight is not None:
        edge_weight = edge_weight[unique_edge_index]
        if is_edge_weight_tensor:
            edge_weight = tf.convert_to_tensor(edge_weight)

    return edge_index, edge_weight


def edge_train_test_split(edge_index, num_nodes, test_size, edge_weight=None, mode="undirected"):
    """

    :param edge_index: edge_index and edge_weight should be directed (i.e. you should have both (a,b) and (b,a))
    :param num_nodes:
    :param test_size:
    :param edge_weight:
    :param mode:
    :return:
    """
    if mode == "undirected":
        is_edge_index_tensor = tf.is_tensor(edge_index)
        is_edge_weight_tensor = tf.is_tensor(edge_weight)

        edge_index = convert_union_to_numpy(edge_index, dtype=np.int32)
        edge_weight = convert_union_to_numpy(edge_weight, dtype=np.float32)

        # row, col = edge_index
        # upper_index = row < col
        # lower_index = row > col
        #
        # if np.sum(upper_index) != np.sum(lower_index):
        #     raise Exception("please convert your graph to a directed graph with 'convert_edge_to_directed'")

        unique_edge_index, unique_edge_weight = extract_unique_edge(edge_index, edge_weight, mode=mode)

        num_unique_edges = unique_edge_index.shape[1]
        train_indices, test_indices = train_test_split(list(range(num_unique_edges)), test_size=test_size)
        train_edge_index = unique_edge_index[:, train_indices]
        test_edge_index = unique_edge_index[:, test_indices]

        if is_edge_index_tensor:
            train_edge_index = tf.convert_to_tensor(train_edge_index)
            test_edge_index = tf.convert_to_tensor(test_edge_index)

        if edge_weight is not None:
            train_edge_weight = unique_edge_weight[train_indices]
            test_edge_weight = unique_edge_weight[test_indices]

            if is_edge_weight_tensor:
                train_edge_weight = tf.convert_to_tensor(train_edge_weight)
                test_edge_weight = tf.convert_to_tensor(test_edge_weight)
        else:
            train_edge_weight = None
            test_edge_weight = None

        return train_edge_index, test_edge_index, train_edge_weight, test_edge_weight

    else:
        raise NotImplementedError()