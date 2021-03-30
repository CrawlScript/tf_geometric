import tensorflow as tf
from tf_geometric.nn.conv.gcn import gcn_mapper
from tf_geometric.utils.graph_utils import remove_self_loop_edge, add_self_loop_edge, get_laplacian, \
    LaplacianMaxEigenvalue
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_updater

CACHE_KEY_CHEBYNET_NORMED_EDGE_TEMPLATE = "chebynet_normed_edge_{}"


def compute_cache_key(normalization_type):
    """
    Compute the cached key based on GCN normalization configurations: renorm and improved

    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :return: The corresponding cached key for the given GCN normalization configuration.
    """
    return CACHE_KEY_CHEBYNET_NORMED_EDGE_TEMPLATE.format(normalization_type)


# def chebynet_compute_lambda_max(edge_index, edge_weight, normalization_type, num_nodes, cache=None):
#     if cache is not None:
#         cache_key = "chebynet_lambda_max_{}".format(normalization_type)
#         cached_data = cache.get(cache_key, None)
#         if cached_data is not None:
#             return cached_data
#
#     lambda_max = LaplacianMaxEigenvalue(edge_index, edge_weight, num_nodes)(normalization_type=normalization_type)
#
#     if cache is not None:
#         cache[cache_key] = lambda_max
#
#     return lambda_max


def chebynet_norm_edge(edge_index, num_nodes, edge_weight=None, normalization_type="sym", use_dynamic_lambda_max=False, cache=None):
    if cache is not None:
        cache_key = compute_cache_key(normalization_type)
        cached_data = cache.get(cache_key, None)
        if cached_data is not None:
            return cached_data

    edge_index, edge_weight = remove_self_loop_edge(edge_index, edge_weight)

    updated_edge_index, updated_edge_weight = get_laplacian(edge_index, num_nodes, edge_weight, normalization_type)

    # lambda_max = chebynet_compute_lambda_max(edge_index, edge_weight, normalization_type, num_nodes, cache=cache)
    if use_dynamic_lambda_max:
        lambda_max = LaplacianMaxEigenvalue(edge_index, num_nodes, edge_weight)(normalization_type=normalization_type)
    else:
        lambda_max = 2.0
    scaled_edge_weight = (2.0 * updated_edge_weight) / lambda_max

    assert edge_weight is not None

    if cache is not None:
        cache[cache_key] = updated_edge_index, scaled_edge_weight

    return updated_edge_index, scaled_edge_weight


def chebynet_cache_normed_edge(graph, normalization_type="sym", use_dynamic_lambda_max=False, override=False):
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
        cache_key = compute_cache_key(normalization_type)
        graph.cache[cache_key] = None
    chebynet_norm_edge(graph.edge_index, graph.num_nodes, graph.edge_weight, normalization_type,
                       use_dynamic_lambda_max=use_dynamic_lambda_max, cache=graph.cache)


def chebynet(x, edge_index, edge_weight, K, kernels, bias=None, activation=None, normalization_type="sym", use_dynamic_lambda_max=False, cache=None):
    num_nodes = tf.shape(x)[0]
    # lambda_max = chebynet_compute_lambda_max(x, edge_index, edge_weight, normalization_type, cache=cache)

    norm_edge_index, norm_edge_weight = chebynet_norm_edge(edge_index, num_nodes, edge_weight, normalization_type,
                                                           use_dynamic_lambda_max=use_dynamic_lambda_max, cache=cache)

    T0_x = x
    T1_x = x
    out = tf.matmul(T0_x, kernels[0])

    if K > 1:
        T1_x = aggregate_neighbors(x, norm_edge_index, norm_edge_weight, gcn_mapper, sum_reducer, identity_updater)
        out += tf.matmul(T1_x, kernels[1])

    for i in range(2, K):
        T2_x = aggregate_neighbors(T1_x, norm_edge_index, norm_edge_weight, gcn_mapper, sum_reducer,
                                   identity_updater)  ##L^T_{k-1}(L^)
        T2_x = 2.0 * T2_x - T0_x
        out += tf.matmul(T2_x, kernels[i])

        T0_x, T1_x = T1_x, T2_x

    if bias is not None:
        out += bias

    if activation is not None:
        out = activation(out)

    return out
