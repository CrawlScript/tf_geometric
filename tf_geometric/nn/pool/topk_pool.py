# coding=utf-8
import tensorflow as tf
from tensorflow import keras

from tf_geometric.utils.union_utils import union_len


def topk_pool(edge_index, edge_score, k=None, ratio=None):
    """

    :param edge_index:
    :param edge_score: 1-D Array
    :param k:
    :param ratio:
    :return: sampled_edge_index, sampled_edge_score, sample_index
    """

    if k is None and ratio is None:
        raise Exception("you should provide either k or ratio for topk_pool")
    elif k is not None and ratio is not None:
        raise Exception("you should provide either k or ratio for topk_pool, not both of them")

    # num_nodes = union_len(node_score)
    num_edges = union_len(edge_index[0])
    edge_ones = tf.ones([num_edges], dtype=tf.int32)
    num_neighbors = tf.math.segment_sum(edge_ones, edge_index[0])
    num_max_neighbors = tf.reduce_max(num_neighbors)

    # max index of source nodes + 1
    num_seen_nodes = num_neighbors.shape[0]

    min_score = tf.reduce_min(edge_score)

    num_edges_before = tf.concat([
        tf.zeros([1], dtype=tf.int32),
        tf.math.cumsum(num_neighbors)[:-1]
    ], axis=0)

    neighbor_index_for_source = tf.range(0, num_edges) - tf.gather(num_edges_before, edge_index[0])

    score_matrix = tf.cast(tf.fill([num_seen_nodes, num_max_neighbors], min_score - 1.0), dtype=tf.float32)
    score_index = tf.stack([edge_index[0], neighbor_index_for_source], axis=1)
    score_matrix = tf.tensor_scatter_nd_update(score_matrix, score_index, edge_score)

    sort_index = tf.argsort(score_matrix, axis=-1, direction="DESCENDING")

    if k is not None:
        node_k = tf.math.minimum(
            tf.cast(tf.fill([num_seen_nodes], k), dtype=tf.int32),
            num_neighbors
        )
    else:
        node_k = tf.cast(
            tf.math.ceil(tf.cast(num_neighbors, dtype=tf.float32) * tf.cast(ratio, dtype=tf.float32)),
            dtype=tf.int32
        )

    left_k_index = [[row_index, col_index]
                    for row_index, num_cols in enumerate(node_k.numpy())
                    for col_index in range(num_cols)]
    left_k_index = tf.convert_to_tensor(left_k_index, dtype=tf.int32)

    sample_col_index = tf.gather_nd(sort_index, left_k_index)
    sample_row_index = left_k_index[:, 0]
    sample_matrix_index = tf.stack([sample_row_index, sample_col_index], axis=1)

    sampled_edge_score = tf.gather_nd(score_matrix, sample_matrix_index)
    sample_index = tf.gather(num_edges_before, sample_row_index) + sample_col_index
    sampled_edge_index = tf.gather(edge_index, sample_index, axis=1)

    return sampled_edge_index, sampled_edge_score, sample_index


