# coding=utf-8
import tensorflow as tf

from tf_geometric.utils.graph_utils import convert_dense_adj_to_edge, remove_self_loop_edge, add_self_loop_edge


def cluster_pool(x, edge_index, edge_weight, assign_edge_index, assign_edge_weight, num_clusters, num_nodes=None):
    """
    Coarsen the input Graph based on cluster assignment of nodes and output pooled Graph.

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param assign_edge_index: Tensor, shape: [2, num_nodes], edge between clusters and nodes, where each edge
        denotes a node belongs to a specific cluster.
    :param assign_edge_weight: Tensor or None, shape: [num_nodes], the corresponding weight for assign_edge_index
    :param num_clusters: Number of clusters.
    :param num_nodes: Number of nodes, Optional, used for boosting performance.
    :return: Pooled Graph: [pooled_x, pooled_edge_index, pooled_edge_weight]
    """

    # Manually passing in num_nodes, num_clusters, and num_graphs can boost the performance
    if num_nodes is None:
        if x is None:
            raise Exception("Please provide num_nodes if x is None")
        else:
            num_nodes = tf.shape(x)[0]

    assign_row, assign_col = assign_edge_index[0], assign_edge_index[1]

    transposed_sparse_assign_probs = tf.SparseTensor(
        indices=tf.cast(tf.stack([assign_col, assign_row], axis=1), dtype=tf.int64),
        values=assign_edge_weight,
        dense_shape=[num_clusters, num_nodes]
    )
    transposed_sparse_assign_probs = tf.sparse.reorder(transposed_sparse_assign_probs)

    sparse_adj = tf.SparseTensor(
        indices=tf.cast(tf.transpose(edge_index, [1, 0]), dtype=tf.int64),
        values=edge_weight,
        dense_shape=[num_nodes, num_nodes]
    )
    sparse_adj = tf.sparse.reorder(sparse_adj)

    # compute S'AS
    # Since tf only support sparse @ dense, we compute S'AS as (S' @ (S' @ dense(A))')'
    pooled_adj = tf.sparse.sparse_dense_matmul(transposed_sparse_assign_probs, tf.sparse.to_dense(sparse_adj))
    pooled_adj = tf.sparse.sparse_dense_matmul(transposed_sparse_assign_probs, tf.transpose(pooled_adj, [1, 0]))
    pooled_adj = tf.transpose(pooled_adj, [1, 0])

    pooled_edge_index, pooled_edge_weight = convert_dense_adj_to_edge(pooled_adj)

    # pooled_edge_index, pooled_edge_weight = remove_self_loop_edge(pooled_edge_index, pooled_edge_weight)

    if x is not None:
        pooled_x = tf.sparse.sparse_dense_matmul(transposed_sparse_assign_probs, x)
    else:
        pooled_x = None
    return pooled_x, pooled_edge_index, pooled_edge_weight
