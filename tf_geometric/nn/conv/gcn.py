# coding=utf-8
import tensorflow as tf
from tf_geometric.sparse.sparse_adj import SparseAdj
import tf_sparse as tfs


# new API
CACHE_KEY_GCN_NORMED_ADJ_TEMPLATE = "gcn_normed_adj_{}_{}"


def compute_cache_key(renorm, improved):
    """
    Compute the cached key based on GCN normalization configurations: renorm and improved

    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :return: The corresponding cached key for the given GCN normalization configuration.
    """
    return CACHE_KEY_GCN_NORMED_ADJ_TEMPLATE.format(renorm, improved)


def gcn_norm_adj(sparse_adj, renorm=True, improved=False, cache: dict = None):
    """
    Compute normed edge (updated edge_index and normalized edge_weight) for GCN normalization.

    :param sparse_adj: SparseAdj, sparse adjacency matrix.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching the updated edge_index and normalized edge_weight.
    :return: Normed edge (updated edge_index and normalized edge_weight).
    """

    if cache is not None:
        cache_key = compute_cache_key(renorm, improved)
        cached_data = cache.get(cache_key, None)
        if cached_data is not None:
            # return cached_data
            return SparseAdj(cached_data[0], cached_data[1], cached_data[2])
        else:
            if not tf.executing_eagerly():
                raise Exception("If you want to use cache inside a tf.function, you should manually build the cache before calling the tf.function")


    fill_weight = 2.0 if improved else 1.0

    if renorm:
        sparse_adj = sparse_adj.add_self_loop(fill_weight=fill_weight)

    deg = sparse_adj.segment_sum(axis=-1)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    # (D^(-1/2)A)D^(-1/2)
    normed_sparse_adj = tfs.sparse_diag_matmul(tfs.diag_sparse_matmul(deg_inv_sqrt, sparse_adj), deg_inv_sqrt)

    if not renorm:
        normed_sparse_adj = normed_sparse_adj.add_self_loop(fill_weight=fill_weight)

    if cache is not None:
        # cache[cache_key] = normed_sparse_adj
        # tf.function will convert numpy arrays as constants, while tensors may be converted into placeholders
        cache[cache_key] = normed_sparse_adj.index.numpy(), normed_sparse_adj.value.numpy(), normed_sparse_adj._shape.numpy()

    return normed_sparse_adj


def gcn_build_cache_for_graph(graph, renorm=True, improved=False, override=False):
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

    sparse_adj = SparseAdj(graph.edge_index, graph.edge_weight, [graph.num_nodes, graph.num_nodes])
    gcn_norm_adj(sparse_adj, renorm, improved, graph.cache)


# old API
def gcn_norm_edge(edge_index, num_nodes, edge_weight=None, renorm=True, improved=False, cache: dict = None):
    """
    Compute normed edge (updated edge_index and normalized edge_weight) for GCN normalization.

    :param edge_index: Tensor, shape: [2, num_edges], edge information.
    :param num_nodes: Number of nodes.
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching the updated edge_index and normalized edge_weight.
    :return: Normed edge (updated edge_index and normalized edge_weight).

    .. deprecated:: 0.0.56
        Use ``gcn_norm_adj`` instead.
    """
    sparse_adj = SparseAdj(edge_index, edge_weight, [num_nodes, num_nodes])
    normed_sparse_adj = gcn_norm_adj(sparse_adj, renorm=renorm, improved=improved, cache=cache)
    return normed_sparse_adj.edge_index, normed_sparse_adj.edge_weight


# old API
def gcn_cache_normed_edge(graph, renorm=True, improved=False, override=False):
    """
    Manually compute the normed edge based on the given GCN normalization configuration (renorm and improved) and put it in graph.cache.
    If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

    :param graph: tfg.Graph, the input graph.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param override: Whether to override existing cached normed edge.
    :return: None

    .. deprecated:: 0.0.56
        Use ``gcn_build_cache_for_graph`` instead.
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

              gcn_layer.build_cache_for_graph(graph)

        - (2) If you're using functional API tfg.nn.gcn:

              from tf_geometric.nn.conv.gcn import gcn_build_cache_for_graph
              gcn_build_cache_for_graph(graph)

    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    num_nodes = tfs.shape(x)[0]

    sparse_adj = SparseAdj(edge_index, edge_weight, [num_nodes, num_nodes])
    normed_sparse_adj = gcn_norm_adj(sparse_adj, renorm, improved, cache)

    # SparseTensor is usually used for one-hot node features (For example, feature-less nodes.)
    if isinstance(x, tf.sparse.SparseTensor):
        h = tf.sparse.sparse_dense_matmul(x, kernel)
    else:
        h = x @ kernel

    h = normed_sparse_adj @ h

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h
