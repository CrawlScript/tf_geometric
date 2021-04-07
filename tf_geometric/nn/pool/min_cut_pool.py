# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.pool.common_pool import sum_pool

from tf_geometric.nn.pool.cluster_pool import cluster_pool
from tf_geometric.utils.graph_utils import convert_dense_assign_to_edge, adj_norm_edge


# compute trace of S'AS
def _compute_STAS_trace(transposed_S, transposed_A, num_clusters):
    # Since tf only support sparse @ dense, we compute S'AS as S' (S' @ dense(A'))
    STAT = tf.sparse.sparse_dense_matmul(transposed_S, tf.sparse.to_dense(transposed_A))
    diag = tf.sparse.reduce_sum(transposed_S * STAT, axis=-1)

    diag = tf.reshape(diag, [-1, num_clusters])
    trace = tf.reduce_sum(diag, axis=-1)
    return trace


def min_cut_pool_compute_loss(edge_index, edge_weight, node_graph_index, dense_assign):
    dense_shape = tf.shape(dense_assign)
    num_nodes = dense_shape[0]
    num_clusters = dense_shape[1]
    num_graphs = tf.reduce_max(node_graph_index) + 1

    edge_index, normed_edge_weight = adj_norm_edge(edge_index, num_nodes, edge_weight, add_self_loop=False)

    row, col = edge_index[0], edge_index[1]
    degree = tf.math.unsorted_segment_sum(normed_edge_weight, row, num_segments=num_nodes)

    assign_edge_index, assign_edge_weight = convert_dense_assign_to_edge(dense_assign, node_graph_index,
                                                                         num_nodes=num_nodes, num_clusters=num_clusters)

    assign_row, assign_col = assign_edge_index[0], assign_edge_index[1]

    transposed_sparse_assign_probs = tf.SparseTensor(
        indices=tf.cast(tf.stack([assign_col, assign_row], axis=1), dtype=tf.int64),
        values=assign_edge_weight,
        dense_shape=[num_clusters * num_graphs, num_nodes]
    )
    transposed_sparse_assign_probs = tf.sparse.reorder(transposed_sparse_assign_probs)

    transposed_sparse_adj = tf.SparseTensor(
        indices=tf.cast(tf.transpose(edge_index, [1, 0]), dtype=tf.int64),
        values=normed_edge_weight,
        dense_shape=[num_nodes, num_nodes]
    )
    transposed_sparse_adj = tf.sparse.reorder(transposed_sparse_adj)

    sparse_degree = tf.SparseTensor(
        indices=tf.stack([tf.range(0, num_nodes, dtype=tf.int64), tf.range(0, num_nodes, dtype=tf.int64)], axis=1),
        values=degree,
        dense_shape=[num_nodes, num_nodes]
    )

    intra_edge_sum = _compute_STAS_trace(transposed_sparse_assign_probs, transposed_sparse_adj, num_clusters)
    all_edge_sum = _compute_STAS_trace(transposed_sparse_assign_probs, sparse_degree, num_clusters)

    cut_losses = -intra_edge_sum / (all_edge_sum + 1e-8)

    sparse_assign_probs = tf.sparse.transpose(transposed_sparse_assign_probs, [1, 0])
    STS = tf.sparse.sparse_dense_matmul(transposed_sparse_assign_probs, tf.sparse.to_dense(sparse_assign_probs))


    STS = STS @ tf.cast(tf.concat([tf.eye(num_clusters) for _ in range(num_graphs)] , axis=0), tf.float32)
    STS = tf.reshape(STS, [-1, num_clusters, num_clusters])
    norm_STS = tf.norm(STS, ord="euclidean", axis=[-2, -1], keepdims=True)
    normed_STS = STS / (norm_STS + 1e-8)
    deviation = normed_STS - tf.eye(tf.shape(normed_STS)[1], batch_shape=[tf.shape(STS)[0]]) / tf.sqrt(tf.cast(num_clusters, tf.float32))
    orth_losses = tf.norm(deviation, ord="euclidean", axis=[-2, -1])
    return cut_losses, orth_losses

