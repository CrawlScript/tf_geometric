# coding=utf-8
import tensorflow as tf

from tf_geometric.utils.graph_utils import convert_dense_adj_to_edge


def diff_pool(x, edge_index, edge_weight, node_graph_index,
              feature_gnn, assign_gnn,
              num_clusters, bias=None, activation=None, training=None):
    """
    Functional API for DiffPool: "Hierarchical graph representation learning with differentiable pooling"

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
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
    """
    num_nodes = tf.shape(x)[0]
    assign_logits = assign_gnn([x, edge_index, edge_weight], training=training)
    assign_probs = tf.nn.softmax(assign_logits, axis=-1)

    assign_row, assign_col = tf.meshgrid(tf.range(num_nodes), tf.range(num_clusters), indexing="ij")
    # multi graphs, the j_th clusters of the i_th graph is assigned a new cluster index: num_clusters * i + j
    assign_col += tf.expand_dims(node_graph_index, axis=-1) * num_clusters

    assign_row = tf.reshape(assign_row, [-1])
    assign_col = tf.reshape(assign_col, [-1])
    assign_probs = tf.reshape(assign_probs, [-1])

    num_graphs = tf.reduce_max(node_graph_index) + 1

    # sparse_assign_probs = tf.SparseTensor(
    #     indices=tf.cast(tf.stack([assign_row, assign_col], axis=1), dtype=tf.int64),
    #     values=assign_probs,
    #     dense_shape=[num_nodes, num_clusters * num_graphs]
    # )

    transposed_sparse_assign_probs = tf.SparseTensor(
        indices=tf.cast(tf.stack([assign_col, assign_row], axis=1), dtype=tf.int64),
        values=assign_probs,
        dense_shape=[num_clusters * num_graphs, num_nodes]
    )
    transposed_sparse_assign_probs = tf.sparse.reorder(transposed_sparse_assign_probs)

    num_edges = tf.shape(edge_index)[-1]

    if edge_weight is None:
        edge_weight = tf.ones(edge_weight, num_edges)

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

    h = feature_gnn([x, edge_index, edge_weight], training=training)
    pooled_h = tf.sparse.sparse_dense_matmul(transposed_sparse_assign_probs, h)

    pooled_node_graph_index = tf.reshape(tf.tile(tf.expand_dims(tf.range(num_graphs), axis=-1), [1, num_clusters]), -1)

    if bias is not None:
        pooled_h += bias

    if activation is not None:
        pooled_h = activation(pooled_h)

    return pooled_h, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index

