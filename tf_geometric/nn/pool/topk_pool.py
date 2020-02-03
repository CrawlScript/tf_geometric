# coding=utf-8
import tensorflow as tf
from tensorflow import keras

from tf_geometric.utils.union_utils import union_len


def topk_pool(source_index, score, k=None, ratio=None):
    """

    :param source_index: index of source node (of edge) or source graph (of node)
    :param score: 1-D Array
    :param k:
    :param ratio:
    :return: sampled_edge_index, sampled_edge_score, sample_index
    """

    if k is None and ratio is None:
        raise Exception("you should provide either k or ratio for topk_pool")
    elif k is not None and ratio is not None:
        raise Exception("you should provide either k or ratio for topk_pool, not both of them")

    score = tf.reshape(score, [-1])

    num_targets = union_len(source_index)
    target_ones = tf.ones([num_targets], dtype=tf.int32)
    num_targets_for_sources = tf.math.segment_sum(target_ones, source_index)
    # number of columns for score matrix
    num_cols = tf.reduce_max(num_targets_for_sources)

    # max index of source + 1
    num_seen_sources = num_targets_for_sources.shape[0]

    min_score = tf.reduce_min(score)

    num_targets_before = tf.concat([
        tf.zeros([1], dtype=tf.int32),
        tf.math.cumsum(num_targets_for_sources)[:-1]
    ], axis=0)

    target_index_for_source = tf.range(0, num_targets) - tf.gather(num_targets_before, source_index)

    score_matrix = tf.cast(tf.fill([num_seen_sources, num_cols], min_score - 1.0), dtype=tf.float32)
    score_index = tf.stack([source_index, target_index_for_source], axis=1)
    score_matrix = tf.tensor_scatter_nd_update(score_matrix, score_index, score)

    sort_index = tf.argsort(score_matrix, axis=-1, direction="DESCENDING")

    if k is not None:
        node_k = tf.math.minimum(
            tf.cast(tf.fill([num_seen_sources], k), dtype=tf.int32),
            num_targets_for_sources
        )
    else:
        node_k = tf.cast(
            tf.math.ceil(tf.cast(num_targets_for_sources, dtype=tf.float32) * tf.cast(ratio, dtype=tf.float32)),
            dtype=tf.int32
        )

    left_k_index = [[row_index, col_index]
                    for row_index, num_cols in enumerate(node_k.numpy())
                    for col_index in range(num_cols)]

    left_k_index = tf.convert_to_tensor(left_k_index, dtype=tf.int32)

    sample_col_index = tf.gather_nd(sort_index, left_k_index)
    sample_row_index = left_k_index[:, 0]

    topk_index = tf.gather(num_targets_before, sample_row_index) + sample_col_index

    return topk_index


