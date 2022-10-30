# coding=utf-8
import tensorflow as tf
# from tf_geometric.sparse.sparse_adj import SparseAdj
import tf_sparse as tfs
from tf_sparse import SparseMatrix


# new API
CACHE_KEY_GCN_NORMED_ADJ_TEMPLATE = "gcn_normed_adj_{}_{}_{}_{}_{}"


def compute_cache_key(norm, add_self_loop, sym, renorm, improved):
    """
    Compute the cached key based on GCN normalization configurations: renorm and improved

    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :return: The corresponding cached key for the given GCN normalization configuration.
    """
    return CACHE_KEY_GCN_NORMED_ADJ_TEMPLATE.format(norm, add_self_loop, sym, renorm, improved)


def _remove_inf_and_nan(x):
    x = tf.where(
        tf.math.logical_or(tf.math.is_inf(x), tf.math.is_nan(x)),
        tf.zeros_like(x),
        x
    )
    return x


def gcn_norm_adj(sparse_adj: SparseMatrix, norm="both", add_self_loop=True, sym=True,
                 renorm=True, improved=False, cache: dict = None):
    """
    Compute normed edge (updated edge_index and normalized edge_weight) for GCN normalization.

    :param sparse_adj: tf_sparse.SparseMatrix, sparse adjacency matrix.
    :param norm: normalization mode both|left|right:
        - both: (D^(-1/2)A)D^(-1/2);
        - left: D^(-1/2)A; 
        - right: AD^(-1/2);
    :param add_self_loop: Whether add self-loop to adj during normalization.
    :param sym: Optional, only used when norm=="both". Setting sym=True indicates that the input
        sparse_adj is symmetric.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching the updated edge_index and normalized edge_weight.
    :return: Normed edge (updated edge_index and normalized edge_weight).
    """

    if cache is not None:
        cache_key = compute_cache_key(norm, add_self_loop, sym, renorm, improved)
        cached_data = cache.get(cache_key, None)
        if cached_data is not None:
            # return cached_data
            return SparseMatrix(cached_data[0], cached_data[1], cached_data[2])
        else:
            if not tf.executing_eagerly():
                raise Exception("If you want to use cache inside a tf.function, you should manually build the cache before calling the tf.function")


    fill_weight = 2.0 if improved else 1.0

    # for non-square adj
    if sparse_adj.shape[0] != sparse_adj.shape[1]:
        if add_self_loop:
            raise Exception("cannot set add_self_loop=True for GCN when sparse_adj.shape[0] != sparse_adj.shape[1]")
        if sym:
            raise Exception("cannot set sym=True for GCN when sparse_adj.shape[0] != sparse_adj.shape[1]")

    if add_self_loop and norm != "both":
        sparse_adj = sparse_adj.add_diag(fill_weight)

    # (D^(-1/2)A)D^(-1/2)
    if norm == "both":
        if add_self_loop and renorm:
            sparse_adj = sparse_adj.add_diag(fill_weight)
            # sparse_adj = sparse_adj.add_self_loop(fill_weight=fill_weight)

        row_deg = sparse_adj.segment_sum(axis=-1)
        row_deg_inv_sqrt = tf.pow(row_deg, -0.5)
        row_deg_inv_sqrt = _remove_inf_and_nan(row_deg_inv_sqrt)
        row_deg_inv_sqrt = tfs.diags(row_deg_inv_sqrt)

        if sym:
            col_deg_inv_sqrt = row_deg_inv_sqrt
        else:
            col_deg = sparse_adj.segment_sum(axis=0)
            col_deg_inv_sqrt = tf.pow(col_deg, -0.5)
            col_deg_inv_sqrt = _remove_inf_and_nan(col_deg_inv_sqrt)
            col_deg_inv_sqrt = tfs.diags(col_deg_inv_sqrt)

        # (D^(-1/2)A)D^(-1/2)
        normed_sparse_adj = row_deg_inv_sqrt @ sparse_adj @ col_deg_inv_sqrt
        # normed_sparse_adj = tfs.sparse_diag_matmul(tfs.diag_sparse_matmul(deg_inv_sqrt, sparse_adj), deg_inv_sqrt)

        if add_self_loop and not renorm:
            normed_sparse_adj = normed_sparse_adj.add_diag(fill_weight)
            # normed_sparse_adj = normed_sparse_adj.add_self_loop(fill_weight=fill_weight)

    # D^(-1/2)A
    elif norm == "left":
        row_deg = sparse_adj.segment_sum(axis=-1)
        row_deg_inv = tf.pow(row_deg, -1)
        row_deg_inv = _remove_inf_and_nan(row_deg_inv)
        row_deg_inv = tfs.diags(row_deg_inv)

        # D^(-1)A
        normed_sparse_adj = row_deg_inv @ sparse_adj

    # AD^(-1/2)
    elif norm == "right":
        col_deg = sparse_adj.segment_sum(axis=-1)
        col_deg_inv = tf.pow(col_deg, -1)
        col_deg_inv = _remove_inf_and_nan(col_deg_inv)
        col_deg_inv = tfs.diags(col_deg_inv)

        # AD^(-1/2)
        normed_sparse_adj = sparse_adj @ col_deg_inv

    else:
        raise Exception("wrong GCN norm type: {}".format(norm))


    if cache is not None:
        # cache[cache_key] = normed_sparse_adj
        # tf.function will convert numpy arrays as constants, while tensors may be converted into placeholders
        cache[cache_key] = normed_sparse_adj.index.numpy(), normed_sparse_adj.value.numpy(), normed_sparse_adj._shape.numpy()

    return normed_sparse_adj


def gcn_build_cache_by_adj(sparse_adj: SparseMatrix, norm="both", add_self_loop=True, sym=True, renorm=True, improved=False, override=False, cache=None):
    """
    Manually compute the normed edge based on the given GCN normalization configuration (renorm and improved) and put it in graph.cache.
    If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

    :param sparse_adj: sparse_adj.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param override: Whether to override existing cached normed edge.
    :return: cache
    """

    if cache is None:
        cache = {}
    elif override:
        cache_key = compute_cache_key(norm, add_self_loop, sym, renorm, improved)
        cache[cache_key] = None

    gcn_norm_adj(sparse_adj, norm, add_self_loop, sym, renorm, improved, cache)
    return cache


def gcn_build_cache_for_graph(graph, norm="both", add_self_loop=True, sym=True, renorm=True, improved=False, override=False):
    """
    Manually compute the normed edge based on the given GCN normalization configuration (renorm and improved) and put it in graph.cache.
    If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

    :param graph: tfg.Graph, the input graph.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param override: Whether to override existing cached normed edge.
    :return: None
    """
    graph.cache = gcn_build_cache_by_adj(graph.adj(),
                                         norm=norm, add_self_loop=add_self_loop, sym=sym,
                                         renorm=renorm, improved=improved, override=override, cache=graph.cache)
    return graph.cache

    # if override:
    #     cache_key = compute_cache_key(renorm, improved)
    #     graph.cache[cache_key] = None
    #
    # sparse_adj = SparseMatrix(graph.edge_index, graph.edge_weight, [graph.num_nodes, graph.num_nodes])
    # gcn_norm_adj(sparse_adj, renorm, improved, graph.cache)


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
    sparse_adj = SparseMatrix(edge_index, edge_weight, [num_nodes, num_nodes])
    normed_sparse_adj = gcn_norm_adj(sparse_adj, renorm=renorm, improved=improved, cache=cache)
    return normed_sparse_adj.index, normed_sparse_adj.value


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


def gcn(x, sparse_adj: SparseMatrix, kernel, bias=None, activation=None,
        norm="both", add_self_loop=True, sym=True,
        renorm=True, improved=False, edge_drop_rate=0.0,
        num_or_size_splits=None,
        training=False, cache=None):
    """
    Functional API for Graph Convolutional Networks.

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param sparse_adj: tf_sparse.SparseMatrix, Adjacency Matrix
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param edge_drop_rate: Dropout rate of the propagation weights.
    :param num_or_size_splits: Split (XW) to compute A(XW) for large graphs (Not affecting the output).
        See the num_or_size_splits param of the tf.split API.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        To use @tf_utils.function with gcn, you should cache the noremd edge information before the first call of the gcn.

        - (1) If you're using OOP APIs tfg.layers.GCN:

              gcn_layer.build_cache_for_graph(graph)

        - (2) If you're using functional API tfg.nn.gcn:

              from tf_geometric.nn.conv.gcn import gcn_build_cache_for_graph
              gcn_build_cache_for_graph(graph)

    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    # num_nodes = tfs.shape(x)[0
    # sparse_adj = SparseMatrix(edge_index, edge_weight, [num_nodes, num_nodes])
    normed_sparse_adj = gcn_norm_adj(sparse_adj, norm=norm, add_self_loop=add_self_loop, sym=sym, renorm=renorm,
                                     improved=improved, cache=cache)
    normed_sparse_adj = normed_sparse_adj.dropout(edge_drop_rate, training=training)

    # SparseTensor is usually used for one-hot node features (For example, feature-less nodes.)

    if kernel is None:
        h = x
    else:
        if isinstance(x, tf.sparse.SparseTensor):
            h = tf.sparse.sparse_dense_matmul(x, kernel)
        else:
            h = x @ kernel

    # if num_or_size_splits is None or 1:
    #     directly compute A(XW), equivalent to:
    #     h = normed_sparse_adj @ h
    # else, split (XW) to compute A(XW) for large graphs:
    #     this does not affect the output, which is also:
    #     h = normed_sparse_adj @ h
    h = normed_sparse_adj.matmul(h, num_or_size_splits=num_or_size_splits)

    # h = normed_sparse_adj @ h

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h
