# coding=utf-8
import tensorflow as tf
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import warnings

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


def convert_edge_to_nx_graph(edge_index, edge_properties=[], convert_to_directed=False):
    edge_index = convert_union_to_numpy(edge_index, dtype=np.int32)
    edge_properties = [convert_union_to_numpy(edge_property) for edge_property in edge_properties]

    g = nx.Graph()
    for i in range(edge_index.shape[1]):
        property_dict = {
        }

        for j, edge_property in enumerate(edge_properties):
            if edge_property is not None:
                property_dict["p_{}".format(j)] = edge_property[i]

        g.add_edge(edge_index[0, i], edge_index[1, i], **property_dict)

    if convert_to_directed:
        g = g.to_directed()

    return g


def convert_edge_to_upper(edge_index, edge_properties=[]):
    edge_index_is_tensor = tf.is_tensor(edge_index)
    edge_properties_is_tensor = [tf.is_tensor(edge_property) for edge_property in edge_properties]

    g = convert_edge_to_nx_graph(edge_index, edge_properties, convert_to_directed=False)

    sorted_edges = [sorted(edge) for edge in g.edges]
    edge_index = np.array(sorted_edges).T

    edge_properties = [
        np.array([item[2] for item in g.edges.data("p_{}".format(i))])
        if edge_property is not None else None
        for i, edge_property in enumerate(edge_properties)
    ]

    if edge_index_is_tensor:
        edge_index = tf.convert_to_tensor(edge_index)

    edge_properties = [
        tf.convert_to_tensor(edge_property) if edge_properties_is_tensor[i] else edge_property
        for i, edge_property in enumerate(edge_properties)
    ]

    return edge_index, edge_properties


# [[1,3,5], [2,1,4]] => [[1,3,5,2,1,4], [2,1,4,1,3,5]]
def convert_edge_to_directed(edge_index, edge_properties=[]):

    edge_index_is_tensor = tf.is_tensor(edge_index)
    edge_properties_is_tensor = [tf.is_tensor(edge_property) for edge_property in edge_properties]

    g = convert_edge_to_nx_graph(edge_index, edge_properties, convert_to_directed=True)

    edge_index = np.array(g.edges).T

    edge_properties = [
        np.array([item[2] for item in g.edges.data("p_{}".format(i))])
        if edge_property is not None else None
        for i, edge_property in enumerate(edge_properties)
    ]

    if edge_index_is_tensor:
        edge_index = tf.convert_to_tensor(edge_index)

    edge_properties = [
        tf.convert_to_tensor(edge_property) if edge_properties_is_tensor[i] else edge_property
        for i, edge_property in enumerate(edge_properties)
    ]

    return edge_index, edge_properties


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


def negative_sampling(num_samples, num_nodes, edge_index=None, replace=True, mode="undirected",
                      batch_size=None):
    """

    :param num_samples:
    :param num_nodes:
    :param edge_index: if edge_index is provided, sampled positive edges will be filtered
    :param replace: only works when edge_index is provided, deciding whether sampled edges should be unique
    :param if batch_size is None, return edge_index, otherwise return a list of batch_size edge_index
    :return:
    """

    edge_index = convert_union_to_numpy(edge_index, np.int32)
    fake_batch_size = 1 if batch_size is None else batch_size

    if edge_index is None:
        sampled_edge_index_list = [np.random.randint(0, num_nodes, [2, num_samples]).astype(np.int32)
                                   for _ in range(fake_batch_size)]
    else:
        if mode == "undirected":
            # fast
            edge_index, _ = convert_edge_to_upper(edge_index)
            adj = np.ones([num_nodes, num_nodes])
            # np.fill_diagonal(adj, 0)
            adj = np.triu(adj, k=1)
            adj[edge_index[0], edge_index[1]] = 0
            neg_edges = np.nonzero(adj)
            neg_edge_index = np.stack(neg_edges, axis=0)
            sampled_edge_index_list = []
            for _ in range(fake_batch_size):
                random_indices = np.random.choice(list(range(neg_edge_index.shape[1])), num_samples, replace=replace)
                sampled_edge_index = neg_edge_index[:, random_indices].astype(np.int32)
                sampled_edge_index_list.append(sampled_edge_index)
        else:
            raise NotImplementedError()

    if tf.is_tensor(edge_index):
        sampled_edge_index_list = [tf.convert_to_tensor(sampled_edge_index)
                                   for sampled_edge_index in sampled_edge_index_list]

    if batch_size is None:
        return sampled_edge_index_list[0]
    else:
        return sampled_edge_index_list


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


def edge_train_test_split(edge_index, test_size, edge_weight=None, mode="undirected", **kwargs):
    """

    :param edge_index:
    :param test_size:
    :param edge_weight:
    :param mode:
    :return:
    """

    # todo: warn user if they pass into "num_nodes", deprecated
    if "num_nodes" in kwargs:
        warnings.warn("argument \"num_nodes\" is deprecated for the method \"edge_train_test_split\", you can remove it")

    if mode == "undirected":
        is_edge_index_tensor = tf.is_tensor(edge_index)
        is_edge_weight_tensor = tf.is_tensor(edge_weight)

        edge_index = convert_union_to_numpy(edge_index, dtype=np.int32)
        edge_weight = convert_union_to_numpy(edge_weight, dtype=np.float32)

        upper_edge_index, [upper_edge_weight] = convert_edge_to_upper(edge_index, [edge_weight])

        num_unique_edges = upper_edge_index.shape[1]
        train_indices, test_indices = train_test_split(list(range(num_unique_edges)), test_size=test_size)
        undirected_train_edge_index = upper_edge_index[:, train_indices]
        undirected_test_edge_index = upper_edge_index[:, test_indices]

        if is_edge_index_tensor:
            undirected_train_edge_index = tf.convert_to_tensor(undirected_train_edge_index)
            undirected_test_edge_index = tf.convert_to_tensor(undirected_test_edge_index)

        if edge_weight is not None:
            undirected_train_edge_weight = upper_edge_weight[train_indices]
            undirected_test_edge_weight = upper_edge_weight[test_indices]

            if is_edge_weight_tensor:
                undirected_train_edge_weight = tf.convert_to_tensor(undirected_train_edge_weight)
                undirected_test_edge_weight = tf.convert_to_tensor(undirected_test_edge_weight)
        else:
            undirected_train_edge_weight = None
            undirected_test_edge_weight = None

        return undirected_train_edge_index, undirected_test_edge_index, undirected_train_edge_weight, undirected_test_edge_weight

    else:
        raise NotImplementedError()


def compute_edge_mask_by_node_index(edge_index, node_index):

    edge_index_is_tensor = tf.is_tensor(edge_index)

    node_index = convert_union_to_numpy(node_index)
    edge_index = convert_union_to_numpy(edge_index)

    max_node_index = np.maximum(np.max(edge_index), np.max(node_index))
    node_mask = np.zeros([max_node_index + 1]).astype(np.bool)
    node_mask[node_index] = True
    row, col = edge_index
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = np.logical_and(row_mask, col_mask)

    if edge_index_is_tensor:
        edge_mask = tf.convert_to_tensor(edge_mask, dtype=tf.bool)

    return edge_mask


