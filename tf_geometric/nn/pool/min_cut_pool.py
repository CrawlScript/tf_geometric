# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.pool.cluster_pool import cluster_pool
from tf_geometric.utils.graph_utils import convert_dense_assign_to_edge, adj_norm_edge, remove_self_loop_edge


# compute trace of S'AS
def _compute_STAS_trace(transposed_S, transposed_A, num_clusters):
    # Since tf only support sparse @ dense, we compute S'AS as S' (S' @ dense(A'))
    STAT = tf.sparse.sparse_dense_matmul(transposed_S, tf.sparse.to_dense(transposed_A))
    diag = tf.sparse.reduce_sum(transposed_S * STAT, axis=-1)

    diag = tf.reshape(diag, [-1, num_clusters])
    trace = tf.reduce_sum(diag, axis=-1)
    return trace


def min_cut_pool_compute_losses(edge_index, edge_weight, node_graph_index, dense_assign, normed_edge_weight=None,
                                cache=None):
    dense_shape = tf.shape(dense_assign)
    num_nodes = dense_shape[0]
    num_clusters = dense_shape[1]
    num_graphs = tf.reduce_max(node_graph_index) + 1

    if normed_edge_weight is None:
        _, normed_edge_weight = adj_norm_edge(edge_index, num_nodes, edge_weight, add_self_loop=False, cache=cache)

    row, col = edge_index[0], edge_index[1]
    degree = tf.math.unsorted_segment_sum(normed_edge_weight, row, num_segments=num_nodes)

    assign_edge_index, assign_edge_weight = convert_dense_assign_to_edge(dense_assign, node_graph_index,
                                                                         num_nodes=num_nodes, num_clusters=num_clusters)

    assign_row, assign_col = assign_edge_index[0], assign_edge_index[1]

    # S': [num_clusters * num_graphs, num_nodes], where num_clusters * num_graphs is the number of unique clusters in a batch
    transposed_sparse_assign_probs = tf.SparseTensor(
        indices=tf.cast(tf.stack([assign_col, assign_row], axis=1), dtype=tf.int64),
        values=assign_edge_weight,
        dense_shape=[num_clusters * num_graphs, num_nodes]
    )
    transposed_sparse_assign_probs = tf.sparse.reorder(transposed_sparse_assign_probs)

    # A': [num_nodes, num_nodes]
    transposed_sparse_adj = tf.SparseTensor(
        # transpose edge_index to obtain A' instead of A
        indices=tf.cast(tf.transpose(edge_index, [1, 0]), dtype=tf.int64),
        values=normed_edge_weight,
        dense_shape=[num_nodes, num_nodes]
    )
    transposed_sparse_adj = tf.sparse.reorder(transposed_sparse_adj)

    # D: [num_nodes, num_nodes]
    transposed_sparse_degree = tf.SparseTensor(
        indices=tf.stack([tf.range(0, num_nodes, dtype=tf.int64), tf.range(0, num_nodes, dtype=tf.int64)], axis=1),
        values=degree,
        dense_shape=[num_nodes, num_nodes]
    )
    transposed_sparse_degree = tf.sparse.reorder(transposed_sparse_degree)

    # cut_loss ------------------------------------------------------------

    # trace(S'AS)
    intra_edge_sum = _compute_STAS_trace(transposed_sparse_assign_probs, transposed_sparse_adj, num_clusters)
    # trace(S'DS)
    all_edge_sum = _compute_STAS_trace(transposed_sparse_assign_probs, transposed_sparse_degree, num_clusters)

    # [num_graphs]
    cut_losses = tf.reduce_mean(-intra_edge_sum / (all_edge_sum + 1e-8))

    # orth_loss ------------------------------------------------------------

    # S: [num_nodes, num_clusters * num_graphs], where num_clusters * num_graphs is the number of unique clusters in a batch
    sparse_assign_probs = tf.sparse.transpose(transposed_sparse_assign_probs, [1, 0])
    # S'S
    STS = tf.sparse.sparse_dense_matmul(transposed_sparse_assign_probs, tf.sparse.to_dense(sparse_assign_probs))
    # [num_clusters * num_graphs, num_clusters * num_graphs] @ [num_clusters * num_graphs, num_clusters] -> [num_clusters * num_graphs, num_clusters]
    STS = STS @ tf.tile(tf.eye(num_clusters, dtype=tf.float32), [num_graphs, 1])
    # [num_graphs, num_clusters, num_clusters]
    STS = tf.reshape(STS, [-1, num_clusters, num_clusters])
    # [num_graphs, 1, 1]
    norm_STS = tf.norm(STS, ord="euclidean", axis=[-2, -1], keepdims=True)
    # [num_graphs, num_clusters, num_clusters]
    normed_STS = STS / (norm_STS + 1e-8)
    # [num_graphs, num_clusters, num_clusters]
    deviation = normed_STS - tf.eye(tf.shape(normed_STS)[1], batch_shape=[tf.shape(STS)[0]]) / tf.sqrt(
        tf.cast(num_clusters, tf.float32))
    # [num_graphs]
    orth_losses = tf.reduce_mean(tf.norm(deviation, ord="euclidean", axis=[-2, -1]))
    return cut_losses, orth_losses


def min_cut_pool_coarsen(x, edge_index, edge_weight, node_graph_index, dense_assign,
                         num_nodes=None, num_clusters=None, num_graphs=None, normed_edge_weight=None, cache=None):
    """
    Coarsening method for MinCutPool: "Spectral Clustering with Graph Neural Networks for Graph Pooling"
    Coarsen the input BatchGraph (graphs) based on cluster assignment of nodes and output pooled BatchGraph (graphs).
    Graphs should be modeled as a BatchGraph like format and each graph has the same number of clusters.

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param node_graph_index: Tensor/NDArray, shape: [num_nodes], graph index for each node
    :param dense_assign: Tensor, [num_nodes, num_clusters], cluster assignment matrix of nodes.
    :param num_nodes: Number of nodes, Optional, used for boosting performance.
    :param num_clusters: Number of clusters, Optional, used for boosting performance.
    :param num_graphs: Number of graphs, Optional, used for boosting performance.
    :return: Pooled BatchGraph: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
    """

    # Manually passing in num_nodes, num_clusters, and num_graphs can boost the performance
    dense_shape = tf.shape(dense_assign)

    if num_nodes is None:
        num_nodes = dense_shape[0]

    if num_clusters is None:
        num_clusters = dense_shape[1]

    if num_graphs is None:
        num_graphs = tf.reduce_max(node_graph_index) + 1

    if edge_weight is None:
        num_edges = tf.shape(edge_index)[-1]
        edge_weight = tf.ones([num_edges], dtype=tf.float32)

    if normed_edge_weight is None:
        _, normed_edge_weight = adj_norm_edge(edge_index, num_nodes, edge_weight, cache=cache)

    assign_edge_index, assign_edge_weight = convert_dense_assign_to_edge(dense_assign, node_graph_index,
                                                                         num_nodes=num_nodes, num_clusters=num_clusters)

    # Coarsen in a large BatchGraph.
    # Here num_clusters is the sum of number of clusters of graphs in the BatchGraph
    pooled_x, pooled_edge_index, pooled_edge_weight = cluster_pool(x, edge_index, normed_edge_weight, assign_edge_index,
                                                                   assign_edge_weight,
                                                                   num_clusters=num_clusters * num_graphs,
                                                                   num_nodes=num_nodes)

    pooled_node_graph_index = tf.reshape(tf.tile(tf.expand_dims(tf.range(num_graphs), axis=-1), [1, num_clusters]), -1)

    # normalize pooled adj
    pooled_edge_index, pooled_edge_weight = remove_self_loop_edge(pooled_edge_index, pooled_edge_weight)
    # pooled_edge_index, pooled_edge_weight = adj_norm_edge(pooled_edge_index, num_nodes, pooled_edge_weight, cache=cache)

    return pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index


def min_cut_pool(x, edge_index, edge_weight, node_graph_index,
                 feature_gnn, assign_gnn,
                 num_clusters, bias=None, activation=None,
                 gnn_use_normed_edge=True,
                 return_loss_func=False, return_losses=False,
                 cache=None, training=None):
    """
    Functional API for MinCutPool: "Spectral Clustering with Graph Neural Networks for Graph Pooling"

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param node_graph_index: Tensor/NDArray, shape: [num_nodes], graph index for each node
    :param feature_gnn: A GNN model to learn pooled node features, [x, edge_index, edge_weight] => updated_x,
        where updated_x corresponds to high-order node features.
    :param assign_gnn: A GNN model to learn cluster assignment for the pooling, [x, edge_index, edge_weight] => updated_x,
        where updated_x corresponds to the cluster assignment matrix.
    :param num_clusters: Number of clusters for pooling.
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param gnn_use_normed_edge: Boolean. Whether to use normalized edge for feature_gnn and assign_gnn.
    :param return_loss_func: Boolean. If True, return (outputs, loss_func), where loss_func is a callable function
        that returns a list of losses.
    :param return_losses: Boolean. If True, return (outputs, losses), where losses is a list of losses.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
    """

    if return_loss_func and return_losses:
        raise Exception("return_loss_func and return_losses cannot be set to True at the same time")

    if edge_weight is None:
        num_edges = tf.shape(edge_index)[-1]
        edge_weight = tf.ones([num_edges], dtype=tf.float32)

    num_nodes = tf.shape(x)[0]
    num_graphs = tf.reduce_max(node_graph_index) + 1

    # if norm_input_edge:
    _, normed_edge_weight = adj_norm_edge(edge_index, num_nodes, edge_weight, add_self_loop=False, cache=cache)

    if gnn_use_normed_edge:
        gnn_edge_weight = normed_edge_weight
    else:
        gnn_edge_weight = edge_weight

    if cache is None:
        assign_logits = assign_gnn([x, edge_index, gnn_edge_weight], training=training)
        h = feature_gnn([x, edge_index, gnn_edge_weight], training=training)
    else:
        assign_logits = assign_gnn([x, edge_index, gnn_edge_weight], training=training, cache=cache)
        h = feature_gnn([x, edge_index, gnn_edge_weight], training=training, cache=cache)

    assign_probs = tf.nn.softmax(assign_logits, axis=-1)

    pooled_h, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index = min_cut_pool_coarsen(
        h, edge_index, edge_weight, node_graph_index, assign_probs,
        num_nodes=num_nodes, num_clusters=num_clusters, num_graphs=num_graphs,
        normed_edge_weight=normed_edge_weight
    )

    if bias is not None:
        pooled_h += bias

    if activation is not None:
        pooled_h = activation(pooled_h)

    outputs = pooled_h, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index

    if return_loss_func or return_losses:
        # callable loss function
        def loss_func():
            return min_cut_pool_compute_losses(edge_index, edge_weight, node_graph_index, assign_probs,
                                               normed_edge_weight=normed_edge_weight, cache=cache)

        if return_loss_func:
            return outputs, loss_func
        else:
            losses = loss_func()
            return outputs, losses
    else:
        return outputs
