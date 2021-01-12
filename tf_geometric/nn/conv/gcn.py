# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_updater
from tf_geometric.utils.graph_utils import add_self_loop_edge


CACHE_KEY_GCN_NORMED_EDGE_TEMPLATE = "gcn_normed_edge_{}_{}"


def compute_cache_key(renorm, improved):
    """
    Compute the cached key based on GCN normalization configurations: renorm and improved

    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :return: The corresponding cached key for the given GCN normalization configuration.
    """
    return CACHE_KEY_GCN_NORMED_EDGE_TEMPLATE.format(renorm, improved)


def gcn_norm_edge(edge_index, num_nodes, edge_weight=None, renorm=True, improved=False, cache: dict=None):
    """
    Compute normed edge (updated edge_index and normalized edge_weight) for GCN normalization.

    :param edge_index: Tensor, shape: [2, num_edges], edge information.
    :param num_nodes: Number of nodes.
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching the updated edge_index and normalized edge_weight.
    :return: Normed edge (updated edge_index and normalized edge_weight).
    """

    if cache is not None:
        cache_key = compute_cache_key(renorm, improved)
        cached_data = cache.get(cache_key, None)
        if cached_data is not None:
            return cached_data

    if edge_weight is None:
        edge_weight = tf.ones([tf.shape(edge_index)[1]], dtype=tf.float32)

    fill_weight = 2.0 if improved else 1.0

    if renorm:
        edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight=edge_weight, fill_weight=fill_weight)

    row, col = edge_index[0], edge_index[1]
    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=num_nodes)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    normed_edge_weight = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

    if not renorm:
        edge_index, normed_edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight=normed_edge_weight,
                                                            fill_weight=fill_weight)

    if cache is not None:
        cache[cache_key] = edge_index, normed_edge_weight

    return edge_index, normed_edge_weight


def gcn_cache_normed_edge(graph, renorm=True, improved=False, override=False):
    """
    Manually compute the normed edge based on the given GCN normalization configuration (renorm and improved) and put it in graph.cache.
    If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

    :param graph: tfg.Graph, the input graph.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param override: Whether to override existing cached normed edge.
    :return: None
    """
    if override:
        cache_key = compute_cache_key(renorm, improved)
        graph.cache[cache_key] = None
    gcn_norm_edge(graph.edge_index, graph.num_nodes, graph.edge_weight, renorm, improved, graph.cache)


def gcn_mapper(repeated_x, neighbor_x, edge_weight=None):
    return neighbor_x * tf.expand_dims(edge_weight, 1)


def gcn(x, edge_index, edge_weight, kernel, bias=None, activation=None,
        renorm=True, improved=False, cache=None):
    """
    Functional API for Graph Convolutional Networks.

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        To use @tf_utils.function with gcn, you should cache the noremd edge information before the first call of the gcn.

        - (1) If you're using OOP APIs tfg.layers.GCN:

              gcn_layer.cache_normed_edge(graph)

        - (2) If you're using functional API tfg.nn.gcn:

              from tf_geometric.nn.conv.gcn import gcn_cache_normed_edge
              gcn_cache_normed_edge(graph)

    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    num_nodes = tf.shape(x)[0]
    updated_edge_index, normed_edge_weight = gcn_norm_edge(edge_index, num_nodes, edge_weight, renorm, improved, cache)

    h = x @ kernel

    h = aggregate_neighbors(
        h, updated_edge_index, normed_edge_weight,
        gcn_mapper,
        sum_reducer,
        identity_updater,
        num_nodes=num_nodes
    )

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h



