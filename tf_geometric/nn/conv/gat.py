# coding=utf-8
import tensorflow as tf

# from tf_geometric.sparse.sparse_adj import SparseAdj
from tf_sparse import SparseMatrix

from tf_geometric.utils.graph_utils import add_self_loop_edge
import tf_sparse as tfs


# follow Transformer-Style Attention
# Attention is all you need
def gat(x, edge_index,
        query_kernel, query_bias, query_activation,
        key_kernel, key_bias, key_activation,
        kernel, bias=None, activation=None, num_heads=1,
        split_value_heads=True, edge_drop_rate=0.0, training=False):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param query_kernel: Tensor, shape: [num_features, num_query_features], weight for Q in attention
    :param query_bias: Tensor, shape: [num_query_features], bias for Q in attention
    :param query_activation: Activation function for Q in attention.
    :param key_kernel: Tensor, shape: [num_features, num_key_features], weight for K in attention
    :param key_bias: Tensor, shape: [num_key_features], bias for K in attention
    :param key_activation: Activation function for K in attention.
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param num_heads: Number of attention heads.
    :param split_value_heads: Boolean. If true, split V as value attention heads, and then concatenate them as output.
        Else, num_heads replicas of V are used as value attention heads, and the mean of them are used as output.
    :param edge_drop_rate: Dropout rate of attention weights.
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    num_nodes = tfs.shape(x)[0]

    # self-attention
    edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]

    x_is_sparse = isinstance(x, tf.sparse.SparseTensor)

    if x_is_sparse:
        Q = tf.sparse.sparse_dense_matmul(x, query_kernel)
    else:
        Q = x @ query_kernel
    Q += query_bias
    if query_activation is not None:
        Q = query_activation(Q)
    Q = tf.gather(Q, row)

    if x_is_sparse:
        K = tf.sparse.sparse_dense_matmul(x, key_kernel)
    else:
        K = x @ key_kernel
    K += key_bias
    if key_activation is not None:
        K = key_activation(K)
    K = tf.gather(K, col)

    if x_is_sparse:
        V = tf.sparse.sparse_dense_matmul(x, kernel)
    else:
        V = x @ kernel

    # xxxxx_ denotes the multi-head style stuff
    Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
    # splited queries and keys are modeled as virtual vertices
    qk_edge_index_ = tf.concat([edge_index + i * num_nodes for i in range(num_heads)], axis=1)

    scale = tf.math.sqrt(tf.cast(tf.shape(Q_)[-1], tf.float32))
    att_score_ = tf.reduce_sum(Q_ * K_, axis=-1) / scale

    # new implementation based on SparseAdj
    num_nodes_ = num_nodes * num_heads
    sparse_att_adj = SparseMatrix(qk_edge_index_, att_score_, [num_nodes_, num_nodes_]) \
        .segment_softmax(axis=-1) \
        .dropout(edge_drop_rate, training=training)

    if split_value_heads:
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
    else:
        V_ = V
        edge_index_ = tf.tile(edge_index, [1, num_heads])
        sparse_att_adj = SparseMatrix(edge_index_, sparse_att_adj.value, [num_nodes, num_nodes])

    h_ = sparse_att_adj @ V_

    # old implementation
    # normed_att_score_ = segment_softmax(att_score_, qk_edge_index_[0], num_nodes * num_heads)
    #
    # if training and drop_rate > 0.0:
    #     normed_att_score_ = tf.compat.v2.nn.dropout(normed_att_score_, drop_rate)
    #
    # if split_value_heads:
    #     V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
    #     edge_index_ = qk_edge_index_
    # else:
    #     V_ = V
    #     edge_index_ = tf.tile(edge_index, [1, num_heads])
    #
    # h_ = aggregate_neighbors(
    #     V_, edge_index_, normed_att_score_,
    #     gcn_mapper,
    #     sum_reducer,
    #     identity_updater
    # )

    if split_value_heads:
        h = tf.concat(tf.split(h_, num_heads, axis=0), axis=-1)
    else:
        h = h_ / num_heads

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h
