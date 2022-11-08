# coding=utf-8
import tensorflow as tf
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import warnings
from tf_geometric.utils.union_utils import convert_union_to_numpy
from scipy.sparse.linalg import eigs, eigsh
import scipy.sparse

from tf_sparse import SparseMatrix


def convert_edge_index_to_edge_hash(edge_index, num_nodes=None):
    edge_index_is_tensor = tf.is_tensor(edge_index)
    num_nodes_is_none = num_nodes is None
    num_nodes_is_tensor = tf.is_tensor(num_nodes)

    edge_index = tf.cast(edge_index, tf.int64)

    # if not edge_index_is_tensor or edge_index.dtype != tf.int64:
    #     edge_index = tf.convert_to_tensor(edge_index, dtype=tf.int64)

    if num_nodes_is_none:
        num_nodes = tf.reduce_max(edge_index) + 1
    else:
        num_nodes = tf.cast(num_nodes, tf.int64)

    row, col = edge_index[0], edge_index[1]

    edge_hash = num_nodes * row + col

    if not edge_index_is_tensor:
        edge_hash = edge_hash.numpy()

    if num_nodes_is_none:
        if not edge_index_is_tensor:
            num_nodes = num_nodes.numpy()
    else:
        if not num_nodes_is_tensor:
            num_nodes = num_nodes.numpy()

    return edge_hash, num_nodes


def convert_edge_hash_to_edge_index(edge_hash, num_nodes):
    edge_hash_is_tensor = tf.is_tensor(edge_hash)

    # if not edge_hash_is_tensor:
    #     edge_hash = tf.convert_to_tensor(edge_hash)

    edge_hash = tf.cast(edge_hash, tf.int64)
    num_nodes = tf.cast(num_nodes, tf.int64)

    row = tf.math.floordiv(edge_hash, num_nodes)
    col = tf.math.floormod(edge_hash, num_nodes)

    edge_index = tf.stack([row, col], axis=0)
    edge_index = tf.cast(edge_index, tf.int32)

    if not edge_hash_is_tensor:
        edge_index = edge_index.numpy()

    return edge_index


def merge_duplicated_edge(edge_index, edge_props=None, merge_modes=None):
    """
    merge_modes: list of merge_mode ("min", "max", "mean", "sum")
    """

    if edge_props is not None and len(edge_props) > 0:
        if merge_modes is None:
            merge_modes = ["sum"] * len(edge_props)
        elif type(merge_modes) is not list:
            raise Exception("type error: merge_modes should be a list of strings")

    # if edge_props is not None and merge_modes is None:
    #     raise Exception("merge_modes is required if edge_props is provided")

    edge_index_is_tensor = tf.is_tensor(edge_index)
    # edge_props_is_tensor = [tf.is_tensor(edge_prop) for edge_prop in edge_props]

    if not edge_index_is_tensor:
        edge_index = tf.convert_to_tensor(edge_index, dtype=tf.int32)

    edge_hash, hash_num_nodes = convert_edge_index_to_edge_hash(edge_index)
    unique_edge_hash, unique_index = tf.unique(edge_hash)

    unique_edge_index = convert_edge_hash_to_edge_index(unique_edge_hash, hash_num_nodes)

    if tf.executing_eagerly() and not edge_index_is_tensor:
        unique_edge_index = unique_edge_index.numpy()

    if edge_props is None:
        unique_edge_props = None
    else:
        unique_edge_props = []
        for edge_prop, merge_mode in zip(edge_props, merge_modes):

            if edge_prop is None:
                unique_edge_prop = None
            else:

                edge_prop_is_tensor = tf.is_tensor(edge_prop)
                edge_prop = tf.convert_to_tensor(edge_prop)

                if merge_mode == "min":
                    merge_func = tf.math.unsorted_segment_min
                elif merge_mode == "max":
                    merge_func = tf.math.unsorted_segment_max
                elif merge_mode == "mean":
                    merge_func = tf.math.unsorted_segment_mean
                elif merge_mode == "sum":
                    merge_func = tf.math.unsorted_segment_sum
                else:
                    raise Exception("wrong merge mode: {}".format(merge_mode))
                unique_edge_prop = merge_func(edge_prop, unique_index, tf.shape(unique_edge_hash)[0])

                if tf.executing_eagerly() and not edge_prop_is_tensor:
                    unique_edge_prop = unique_edge_prop.numpy()

            unique_edge_props.append(unique_edge_prop)

    return unique_edge_index, unique_edge_props


def convert_edge_to_upper(edge_index, edge_props=None, merge_modes=None):
    """

    :param edge_index:
    :param edge_props:
    :param merge_modes: List of merge modes. Merge Modes: "min" | "max" | "mean" | "sum"
    :return:
    """

    edge_index_is_tensor = tf.is_tensor(edge_index)

    if not edge_index_is_tensor:
        edge_index = tf.convert_to_tensor(edge_index, dtype=tf.int32)

    row = tf.math.reduce_min(edge_index, axis=0)
    col = tf.math.reduce_max(edge_index, axis=0)

    upper_edge_index = tf.stack([row, col], axis=0)
    upper_edge_index, upper_edge_props = merge_duplicated_edge(upper_edge_index, edge_props, merge_modes)

    if tf.executing_eagerly() and not edge_index_is_tensor:
        upper_edge_index = upper_edge_index.numpy()

    return upper_edge_index, upper_edge_props


# [[1,3,5], [2,1,4]] => [[1,3,5,2,1,4], [2,1,4,1,3,5]]
def convert_edge_to_directed(edge_index, edge_props=None, merge_modes=None):
    """
    Convert edge from undirected format to directed format.
    For example, [[1,3,5], [2,1,4]] => [[1,3,5,2,1,4], [2,1,4,1,3,5]]

    :param edge_index: Input edge index.
    :param edge_props: List of edge properties, for example: [edge_weight]
    :param merge_modes: List of merge modes. Merge Modes: "min" | "max" | "mean" | "sum"
    :return:
    """

    edge_index_is_tensor = tf.is_tensor(edge_index)

    if not edge_index_is_tensor:
        edge_index = tf.convert_to_tensor(edge_index, dtype=tf.int32)

    # if edge_props is None:
    #     edge_props = []
    #
    # if merge_modes is None:
    #     merge_modes = []

    if edge_props is not None and len(edge_props) > 0:
        if merge_modes is None:
            merge_modes = ["sum"] * len(edge_props)

    upper_edge_index, upper_edge_props = convert_edge_to_upper(edge_index, edge_props, merge_modes)
    non_self_loop_mask = tf.not_equal(upper_edge_index[0], upper_edge_index[1])

    if tf.reduce_any(non_self_loop_mask):
        non_self_loop_edge_index = tf.boolean_mask(upper_edge_index, non_self_loop_mask, axis=1)
        pure_lower_edge_index = tf.stack([
            non_self_loop_edge_index[1],
            non_self_loop_edge_index[0]
        ], axis=0)
        updated_edge_index = tf.concat([upper_edge_index, pure_lower_edge_index], axis=1)

        if edge_props is None:
            updated_edge_props = None
        else:
            updated_edge_props = []
            for edge_prop, upper_edge_prop, merge_mode in zip(edge_props, upper_edge_props, merge_modes):
                if edge_prop is None:
                    updated_edge_prop = None
                else:
                    pure_lower_edge_prop = tf.boolean_mask(upper_edge_prop, non_self_loop_mask)
                    updated_edge_prop = tf.concat([upper_edge_prop, pure_lower_edge_prop], axis=0)
                    if tf.executing_eagerly() and not tf.is_tensor(edge_prop):
                        updated_edge_prop = updated_edge_prop.numpy()
                updated_edge_props.append(updated_edge_prop)
    else:
        updated_edge_index = edge_index
        updated_edge_props = edge_props

    if tf.executing_eagerly() and not edge_index_is_tensor:
        updated_edge_index = updated_edge_index.numpy()

    return updated_edge_index, updated_edge_props


def convert_x_to_3d(x, source_index, k=None, pad=True):
    source_index_perm = tf.argsort(source_index, stable=True)
    sorted_source_index = tf.gather(source_index, source_index_perm)
    permed_x = tf.gather(x, source_index_perm)

    num_targets = tf.shape(sorted_source_index)[0]
    target_ones = tf.ones([num_targets], dtype=tf.int32)
    num_targets_for_sources = tf.math.segment_sum(target_ones, sorted_source_index)
    max_num_targets_for_sources = tf.reduce_max(num_targets_for_sources)

    # max index of source + 1
    num_seen_sources = tf.shape(num_targets_for_sources)[0]

    num_targets_before = tf.concat([
        tf.zeros([1], dtype=tf.int32),
        tf.math.cumsum(num_targets_for_sources)[:-1]
    ], axis=0)

    target_index_for_source = tf.range(0, num_targets) - tf.gather(num_targets_before, sorted_source_index)

    if k is None:
        k = max_num_targets_for_sources
    elif k > max_num_targets_for_sources:
        if not pad:
            k = max_num_targets_for_sources
    elif k < max_num_targets_for_sources:
        mask = tf.less(target_index_for_source, k)
        target_index_for_source = tf.boolean_mask(target_index_for_source, mask)
        sorted_source_index = tf.boolean_mask(sorted_source_index, mask)
        permed_x = tf.boolean_mask(permed_x, mask)

    h = tf.zeros([num_seen_sources, k, tf.shape(x)[-1]], dtype=x.dtype)
    index = tf.stack([sorted_source_index, target_index_for_source], axis=1)
    h = tf.tensor_scatter_nd_update(h, index, permed_x)
    return h


def remove_self_loop_edge(edge_index, edge_weight=None):
    edge_index_is_tensor = tf.is_tensor(edge_index)
    edge_weight_is_tensor = tf.is_tensor(edge_weight)

    row, col = edge_index[0], edge_index[1]
    mask = tf.not_equal(row, col)

    edge_index = tf.boolean_mask(edge_index, mask, axis=1)
    if edge_weight is not None:
        edge_weight = tf.boolean_mask(edge_weight, mask, axis=0)

    if tf.executing_eagerly() and not edge_index_is_tensor:
        edge_index = edge_index.numpy()

    if tf.executing_eagerly() and edge_weight is not None and not edge_weight_is_tensor:
        edge_weight = edge_weight.numpy()

    return edge_index, edge_weight


def convert_dense_adj_to_edge(dense_adj):
    num_nodes = tf.shape(dense_adj)[0]
    row, col = tf.meshgrid(tf.range(num_nodes), tf.range(num_nodes), indexing="ij")
    row = tf.reshape(row, [-1])
    col = tf.reshape(col, [-1])
    edge_index = tf.stack([row, col], axis=0)
    edge_weight = tf.reshape(dense_adj, [-1])

    edge_mask = tf.not_equal(edge_weight, 0.0)
    edge_index = tf.boolean_mask(edge_index, edge_mask, axis=1)
    edge_weight = tf.boolean_mask(edge_weight, edge_mask)

    return edge_index, edge_weight


def convert_dense_assign_to_edge(dense_assign, node_graph_index=None, num_nodes=None, num_clusters=None):
    """
    Convert a dense assignment matrix [num_nodes, num_clusters] of a Graph or BatchGraph to edge_index, edge_weight.
    In the single-graph scenario (when node_graph_index is None), there are num_clusters clusters and 
        the j_th column of each row correspond to the j_th clusters.
    In the multi-graph scenario (where node_graph_index is not None), there are num_clusters * num_graphs clusters and
        the j_th column of the i_th row correspond to the j_th cluster in the corresponding graph, and the cluster index
        is num_clusters * node_graph_index[i] + j
    
    :param dense_assign: 
    :param node_graph_index: 
    :param num_nodes: 
    :param num_clusters: 
    :return: 
    """
    dense_shape = tf.shape(dense_assign)

    # manually passing in num_row and num_col can boost the computation
    if num_nodes is None:
        num_nodes = dense_shape[0]
    if num_clusters is None:
        num_clusters = dense_shape[1]

    row, col = tf.meshgrid(tf.range(num_nodes), tf.range(num_clusters), indexing="ij")

    # multi graphs, the j_th clusters of the i_th graph is assigned a new cluster index: num_clusters * i + j
    if node_graph_index is not None:
        col += tf.expand_dims(node_graph_index, axis=-1) * num_clusters

    row = tf.reshape(row, [-1])
    col = tf.reshape(col, [-1])
    edge_index = tf.stack([row, col], axis=0)

    edge_weight = tf.reshape(dense_assign, [-1])

    return edge_index, edge_weight


def convert_edge_to_nx_graph(edge_index, edge_properties=None, convert_to_directed=False):
    edge_index = convert_union_to_numpy(edge_index, dtype=np.int32)

    if edge_properties is None:
        edge_properties = []
    else:
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


def add_self_loop_edge(edge_index, num_nodes, edge_weight=None, fill_weight=1.0):
    diagnal_edge_index = tf.stack([tf.range(num_nodes, dtype=tf.int32)] * 2, axis=0)
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
        warnings.warn(
            "argument \"num_nodes\" is deprecated for the method \"edge_train_test_split\", you can remove it")

    if mode == "undirected":
        is_edge_index_tensor = tf.is_tensor(edge_index)
        is_edge_weight_tensor = tf.is_tensor(edge_weight)

        edge_index = convert_union_to_numpy(edge_index, dtype=np.int32)
        edge_weight = convert_union_to_numpy(edge_weight, dtype=np.float32)

        upper_edge_index, [upper_edge_weight] = convert_edge_to_upper(edge_index, [edge_weight], merge_modes=["max"])

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

    max_node_index = tf.maximum(tf.reduce_max(edge_index), tf.reduce_max(node_index))
    node_mask = tf.scatter_nd(tf.expand_dims(node_index, axis=-1), tf.ones_like(node_index), [max_node_index + 1])
    node_mask = tf.cast(node_mask, tf.bool)
    row, col = edge_index[0], edge_index[1]
    row_mask = tf.gather(node_mask, row)
    col_mask = tf.gather(node_mask, col)
    edge_mask = tf.logical_and(row_mask, col_mask)

    if not edge_index_is_tensor:
        edge_mask = edge_mask.numpy()
    return edge_mask


def get_laplacian(edge_index, num_nodes, edge_weight, normalization_type, fill_weight=1.0):
    if normalization_type is not None:
        assert normalization_type in [None, 'sym', 'rw']

    row, col = edge_index[0], edge_index[1]
    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=num_nodes)
    ##L = D - A
    if normalization_type is None:
        edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight, fill_weight=fill_weight)
        row, col = edge_index
        deg_inv = tf.where(
            tf.math.logical_or(tf.math.is_inf(deg), tf.math.is_nan(deg)),
            tf.zeros_like(deg),
            deg
        )
        edge_weight = tf.gather(deg_inv, row) - edge_weight

    ## L^ = D^{-1/2}LD^{-1/2}
    elif normalization_type == 'sym':
        deg_inv_sqrt = tf.pow(deg, -0.5)
        deg_inv_sqrt = tf.where(
            tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
            tf.zeros_like(deg_inv_sqrt),
            deg_inv_sqrt
        )

        normed_edge_weight = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)
        edge_index, tmp = add_self_loop_edge(edge_index, num_nodes, edge_weight=normed_edge_weight,
                                             fill_weight=fill_weight)

        assert tmp is not None
        edge_weight = tmp
    ##L^ = D^{-1}L
    else:
        deg_inv = 1.0 / deg
        deg_inv = tf.where(
            tf.math.logical_or(tf.math.is_inf(deg_inv), tf.math.is_nan(deg_inv)),
            tf.zeros_like(deg_inv),
            deg_inv
        )

        normed_edge_weight = tf.gather(deg_inv, row) * edge_weight

        edge_index, tmp = add_self_loop_edge(edge_index, num_nodes, edge_weight=normed_edge_weight,
                                             fill_weight=fill_weight)

        assert tmp is not None
        edge_weight = tmp

    return edge_index, edge_weight


def to_scipy_sparse_matrix(edge_index, edge_weight=None, num_nodes=None):
    r"""Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    row, col = edge_index

    if edge_weight is None:
        edge_weight = tf.ones(row.shape[0])
    else:
        edge_weight = tf.reshape(edge_weight, [-1])
        assert edge_weight.shape[0] == row.shape[0]

    N = num_nodes
    out = scipy.sparse.coo_matrix((edge_weight, (row, col)), (N, N))
    return out


class RandomNeighborSampler(object):
    def __init__(self, edge_index, edge_weight=None):

        edge_index = convert_union_to_numpy(edge_index, np.int32)
        if edge_weight is not None:
            edge_weight = convert_union_to_numpy(edge_weight)
        else:
            edge_weight = np.ones([edge_index.shape[1]], dtype=np.float32)


        self.num_row_nodes = convert_union_to_numpy(edge_index[0].max() + 1)
        self.num_col_nodes = convert_union_to_numpy(edge_index[1].max() + 1)


        raw_neighbor_dict = {}

        for (a, b), weight in zip(edge_index.T, edge_weight):

            if a not in raw_neighbor_dict:
                neighbors = [[],  []]
                raw_neighbor_dict[a] = neighbors
            else:
                neighbors = raw_neighbor_dict[a]
            neighbors[0].append(b)
            neighbors[1].append(weight)
            # neighbors.append((b, weight))

        self.neighbor_dict = {
            source: [np.array(neighbors[0]), np.array(neighbors[1])]
            for source, neighbors in raw_neighbor_dict.items()
        }

        self.num_sources = len(self.neighbor_dict)
        self.source_index = sorted(self.neighbor_dict.keys())
        self.num_neighbors_dict = {node_index: len(neighbors[0]) for node_index, neighbors in self.neighbor_dict.items()}


    def sample(self, k=None, ratio=None, sampled_node_index=None, padding=False):
        # if k is None and ratio is None:
        #     raise Exception("you should provide either k or ratio")
        # elif k is not None and ratio is not None:
        #     raise Exception("you should provide either k or ratio, not both of them")

        if k is not None and ratio is not None:
            raise Exception("k and ratio cannot be provided simultaneously")

        if k is None and ratio is None:
            sample_all = True
        else:
            sample_all = False


        if sampled_node_index is None:
            # use_virtual_node_index = False

            sampled_row_index = self.source_index
            row_virtual_index = sampled_row_index

            # num_sampled_rows = self.num_row_nodes
            # num_sampled_cols = self.num_col_nodes

        else:
            # use_virtual_node_index = True
            if isinstance(sampled_node_index, tuple):
                sampled_row_index, sampled_col_index = sampled_node_index
            else:
                sampled_row_index = sampled_node_index
                sampled_col_index = sampled_node_index

            row_virtual_mapping = -np.ones([self.num_row_nodes], dtype=np.int64)
            row_virtual_index = np.arange(0, len(sampled_row_index))
            row_virtual_mapping[sampled_row_index] = row_virtual_index

            if isinstance(sampled_node_index, tuple):
                col_virtual_mapping = -np.ones([self.num_col_nodes], dtype=np.int64)
                col_virtual_index = np.arange(0, len(sampled_col_index))
                col_virtual_mapping[sampled_col_index] = col_virtual_index
            else:
                col_virtual_index = row_virtual_index
                col_virtual_mapping = row_virtual_mapping

            # num_sampled_rows = len(sampled_row_index)
            # num_sampled_cols = len(sampled_col_index)


        # num_central_nodes = len(central_node_index)

        sampled_virtual_edge_index_list = []
        sampled_virtual_edge_weight_list = []

        for row_virtual_i, row_i in zip(row_virtual_index, sampled_row_index):
            if row_i not in self.neighbor_dict:
                continue

            neighbor_index, neighbor_weight = self.neighbor_dict[row_i]

            if sampled_node_index is not None:
                virtual_neighbor_index = col_virtual_mapping[neighbor_index]
                mask = virtual_neighbor_index >= 0
                virtual_neighbor_index = virtual_neighbor_index[mask]

                if len(virtual_neighbor_index) == 0:
                    continue

                virtual_neighbor_weight = neighbor_weight[mask]
            else:
                virtual_neighbor_index = neighbor_index
                virtual_neighbor_weight = neighbor_weight


            if sample_all or (ratio is None and not padding and k >= len(virtual_neighbor_index)):
                sampled_virtual_neighbor_index = virtual_neighbor_index
                sampled_virtual_neighbor_weight = virtual_neighbor_weight

            else:
                if ratio is None:
                    num_sampled_neighbors = k
                    replace = padding and k >= len(virtual_neighbor_index)
                else:
                    num_sampled_neighbors = np.ceil(len(virtual_neighbor_index) * ratio).astype(np.int32)
                    replace = False

                range_index = np.arange(0, len(virtual_neighbor_index))
                sampled_index = np.random.choice(range_index, num_sampled_neighbors, replace=replace)

                # print(neighbor_index)

                sampled_virtual_neighbor_index = virtual_neighbor_index[sampled_index]
                sampled_virtual_neighbor_weight = virtual_neighbor_weight[sampled_index]

            sampled_virtual_edge_index = np.stack([np.full(sampled_virtual_neighbor_index.shape, fill_value=row_virtual_i), sampled_virtual_neighbor_index], axis=0)

            sampled_virtual_edge_index_list.append(sampled_virtual_edge_index)
            sampled_virtual_edge_weight_list.append(sampled_virtual_neighbor_weight)

        if len(sampled_virtual_edge_index_list) > 0:
            sampled_virtual_edge_index = np.concatenate(sampled_virtual_edge_index_list, axis=1)
            sampled_virtual_edge_weight = np.concatenate(sampled_virtual_edge_weight_list, axis=0)
        else:
            sampled_virtual_edge_index = None
            sampled_virtual_edge_weight = None

        return sampled_virtual_edge_index, sampled_virtual_edge_weight

                # sampled_neighbor_index = np.random.choice(len(neighbors), num_sampled_neighbors, replace=replace)
                # sampled_neighbors = [neighbors[i] for i in sampled_neighbor_index]

            # for (j, weight) in sampled_neighbors:
            #     sampled_edge_index.append([central, j])
            #     sampled_edge_weight.append(weight)

        # sampled_edge_index = np.array(sampled_edge_index).T
        # sampled_edge_weight = np.array(sampled_edge_weight)

        # sampled_virtual_adj = SparseMatrix(
        #     sampled_virtual_edge_index, 
        #     value=sampled_virtual_edge_weight,
        #     shape=[num_sampled_rows, num_sampled_cols])

        # return sampled_virtual_adj

        # sampled_node_index = np.unique(np.reshape(sampled_edge_index, [-1]))
        # central_node_index_set = {i for i in central_node_index}
        # non_central_node_index = [i for i in sampled_node_index if i not in central_node_index_set]
        # sampled_node_index = np.concatenate([central_node_index, non_central_node_index], axis=0)
        #
        # sampled_edge = sampled_edge_index, sampled_edge_weight
        # if not central_node_index_is_none:
        #     node_index = sampled_node_index, central_node_index, non_central_node_index
        #     return sampled_edge, node_index
        # else:
        #     return sampled_edge


class UniformNeighborSampler(object):
    def __init__(self, edge_index, edge_weight=None):


        edge_index = tf.convert_to_tensor(edge_index, dtype=tf.int32)

        num_edges = tf.shape(edge_index)[1]

        if edge_weight is not None:
            edge_weight = tf.convert_to_tensor(edge_weight, dtype=tf.float32)
        else:
            edge_weight = tf.ones([num_edges], dtype=tf.float32)

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_edges = num_edges

        self.num_row_nodes = tf.reduce_max(edge_index[0])+ 1
        self.num_col_nodes = tf.reduce_max(edge_index[1]) + 1

    def _create_virtual_mapping(self, sampled_index, num_nodes):
        virtual_mapping = -tf.ones([num_nodes], dtype=tf.int32)
        virtual_index = tf.range(0, tf.shape(sampled_index)[0])
        virtual_mapping = tf.tensor_scatter_nd_update(
            virtual_mapping, 
            tf.expand_dims(sampled_index, axis=-1), 
            virtual_index
        )
        return virtual_mapping

    def sample(self, prob, sampled_node_index=None):
        # if k is None and ratio is None:
        #     raise Exception("you should provide either k or ratio")
        # elif k is not None and ratio is not None:
        #     raise Exception("you should provide either k or ratio, not both of them")


        if sampled_node_index is None:
            random_score = tf.random.uniform([self.num_edges], 0.0, 1.0, dtype=tf.float32)
            sample_mask = random_score <= prob
            sampled_edge_index = tf.boolean_mask(self.edge_index, sample_mask, axis=1)
            sampled_edge_weight = tf.boolean_mask(self.edge_weight, sample_mask)
            return sampled_edge_index, sampled_edge_weight
        else:
            # use_virtual_node_index = True
            if isinstance(sampled_node_index, tuple):
                sampled_row_index, sampled_col_index = sampled_node_index
            else:
                sampled_row_index = sampled_node_index
                sampled_col_index = sampled_node_index

            row_virtual_mapping = self._create_virtual_mapping(sampled_row_index, self.num_row_nodes)

            if isinstance(sampled_node_index, tuple):
                col_virtual_mapping = self._create_virtual_mapping(sampled_col_index, self.num_col_nodes)
            else:
                col_virtual_mapping = row_virtual_mapping

            
            row, col = self.edge_index[0], self.edge_index[1]
            virtual_row_ = tf.gather(row_virtual_mapping, row)
            virtual_col_ = tf.gather(col_virtual_mapping, col)
            virtual_mask = tf.math.logical_and(
                virtual_row_>=0,
                virtual_col_>=0
            )
            virtual_row = tf.boolean_mask(virtual_row_, virtual_mask)
            virtual_col = tf.boolean_mask(virtual_col_, virtual_mask)
            virtual_edge_index = tf.stack([virtual_row, virtual_col], axis=0)
            virtual_edge_weight = tf.boolean_mask(self.edge_weight, virtual_mask)

            num_virtual_edges = tf.shape(virtual_edge_index)[1]
            random_score = tf.random.uniform([num_virtual_edges], 0.0, 1.0, dtype=tf.float32)
            sample_mask = random_score <= prob
            sampled_virtual_edge_index = tf.boolean_mask(virtual_edge_index, sample_mask, axis=1)
            sampled_virtual_edge_weight = tf.boolean_mask(virtual_edge_weight, sample_mask)
            return sampled_virtual_edge_index, sampled_virtual_edge_weight



class LaplacianMaxEigenvalue(object):
    def __init__(self, edge_index, num_nodes, edge_weight, is_undirected=True):
        self.num_nodes = num_nodes
        self.edge_index = convert_union_to_numpy(edge_index, np.int32)
        if edge_weight is not None:
            self.edge_weight = convert_union_to_numpy(edge_weight)
        else:
            self.edge_weight = np.ones([self.edge_index.shape[1]], dtype=np.float32)
        self.is_undirected = is_undirected

    def __call__(self, normalization_type='sym'):
        assert normalization_type in [None, 'sym', 'rw']

        edge_index, edge_weight = remove_self_loop_edge(self.edge_index, self.edge_weight)

        edge_index, edge_weight = get_laplacian(self.edge_index, self.num_nodes, edge_weight, normalization_type)

        L = to_scipy_sparse_matrix(edge_index, edge_weight, self.num_nodes)

        eig_fn = eigs
        if self.is_undirected and normalization_type:
            eig_fn = eigsh

        lambda_max = eig_fn(L, k=1, which='LM', return_eigenvectors=False)

        return float(lambda_max.real)


# CACHE_KEY_ADJ_NORMED_EDGE = "adj_normed_edge"

def adj_norm_edge(edge_index, num_nodes, edge_weight=None, add_self_loop=False, cache=None):
    cache_key = "adj_normed_edge"
    if cache is not None:
        cached_data = cache.get(cache_key, None)
        if cached_data is not None:
            return cached_data

    if edge_weight is None:
        edge_weight = tf.ones([tf.shape(edge_index)[1]], dtype=tf.float32)

    if add_self_loop:
        fill_weight = 1.0
        edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight=edge_weight,
                                                     fill_weight=fill_weight)

    row, col = edge_index[0], edge_index[1]
    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=num_nodes)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    normed_edge_weight = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

    if cache is not None:
        cache[cache_key] = edge_index, normed_edge_weight

    return edge_index, normed_edge_weight


def reindex_sampled_edge_index(sampled_edge_index, sampled_node_index):
    sampled_edge_index_is_tensor = tf.is_tensor(sampled_edge_index)
    sampled_node_index_is_tensor = tf.is_tensor(sampled_node_index)

    if not sampled_edge_index_is_tensor:
        sampled_edge_index = tf.convert_to_tensor(sampled_edge_index)

    if not sampled_node_index_is_tensor:
        sampled_node_index = tf.convert_to_tensor(sampled_node_index)

    sampled_node_index = tf.cast(sampled_node_index, sampled_edge_index.dtype)

    node_index_dict = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=sampled_node_index,
            values=tf.range(0, tf.shape(sampled_node_index)[0])
        ),
        default_value=-1
    )
    row, col = sampled_edge_index[0], sampled_edge_index[1]
    reindexed_row = node_index_dict[row]
    reindexed_col = node_index_dict[col]
    reindexed_edge_index = tf.stack([reindexed_row, reindexed_col], axis=0)

    if not sampled_edge_index_is_tensor:
        reindexed_edge_index = reindexed_edge_index.numpy()

    return reindexed_edge_index
