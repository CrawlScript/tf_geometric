# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_updater, sum_reducer, identity_updater
from tf_geometric.nn.kernel.segment import segment_softmax
from tf_geometric.nn.conv.gcn import gcn_mapper


# follow Transformer-Style Attention
# Attention is all you need
from tf_geometric.utils.graph_utils import add_self_loop_edge


def gat(x, edge_index,
        query_kernel, query_bias, query_activation,
        key_kernel, key_bias, key_activation,
        kernel, bias=None, activation=None, num_heads=1, drop_rate=0.0, training=False):
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
    :param drop_rate: Dropout rate.
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    num_nodes = x.shape[0]

    # self-attention
    edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes)

    row, col = edge_index

    Q = query_activation(x @ query_kernel + query_bias)
    Q = tf.gather(Q, row)

    K = key_activation(x @ key_kernel + key_bias)
    K = tf.gather(K, col)

    V = x @ kernel

    # xxxxx_ denotes the multi-head style stuff
    Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
    edge_index_ = tf.concat([edge_index + i * num_nodes for i in range(num_heads)], axis=1)

    att_score_ = tf.reduce_sum(Q_ * K_, axis=-1)
    normed_att_score_ = segment_softmax(att_score_, edge_index_[0], num_nodes * num_heads)

    if training and drop_rate > 0.0:
        normed_att_score_ = tf.compat.v2.nn.dropout(normed_att_score_, drop_rate)

    h_ = aggregate_neighbors(
        V_, edge_index_, normed_att_score_,
        gcn_mapper,
        sum_reducer,
        identity_updater
    )

    h = tf.concat(tf.split(h_, num_heads, axis=0), axis=-1)

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h







