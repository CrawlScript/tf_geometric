# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_updater, sum_reducer
from tf_geometric.nn.kernel.segment import segment_softmax
from tf_geometric.nn.gnn.gcn import gcn_mapper


def gat_mapper(repeated_x, neighbor_x, edge_weight=None):
    return neighbor_x * tf.expand_dims(edge_weight, 1)


# follow Transformer-Style Attention
# Attention is all you need
def gat(x, edge_index, edge_weight,
        query_w, query_b, query_activation,
        key_w, key_b, key_activation,
        dense_w, dense_b=None, activation=None):

    row, col = edge_index

    query = tf.gather(x, row) @ query_w + query_b
    query = query_activation(query)

    key = tf.gather(x, col) @ key_w + key_b
    key = key_activation(key)

    att_score = tf.reduce_sum(query * key, axis=-1)
    normed_att_score = segment_softmax(att_score, row, len(x))

    value = x @ dense_w

    h = aggregate_neighbors(
        value, edge_index, normed_att_score,
        gcn_mapper,
        sum_reducer,
        sum_updater
    )

    if dense_b is not None:
        h += dense_b

    if activation is not None:
        h = activation(h)

    return h







