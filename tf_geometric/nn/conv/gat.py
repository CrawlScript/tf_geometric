# coding=utf-8
import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_updater, sum_reducer, identity_updater
from tf_geometric.nn.kernel.segment import segment_softmax
from tf_geometric.nn.conv.gcn import gcn_mapper


# follow Transformer-Style Attention
# Attention is all you need
from tf_geometric.utils.graph_utils import add_diagonal_edge_index


def gat(x, edge_index,
        query_kernel, query_bias, query_activation,
        key_kernel, key_bias, key_activation,
        kernel, bias=None, activation=None):

    # self-attention
    edge_index, edge_weight = add_diagonal_edge_index(edge_index, len(x))

    row, col = edge_index

    query = tf.gather(x, row) @ query_kernel + query_bias
    query = query_activation(query)

    key = tf.gather(x, col) @ key_kernel + key_bias
    key = key_activation(key)

    att_score = tf.reduce_sum(query * key, axis=-1)
    normed_att_score = segment_softmax(att_score, row, len(x))

    value = x @ kernel

    h = aggregate_neighbors(
        value, edge_index, normed_att_score,
        gcn_mapper,
        sum_reducer,
        identity_updater
    )

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h







