# coding=utf-8
import tensorflow as tf
from tf_geometric.utils.union_utils import union_len


def topk_pool(source_index, score, k=None, ratio=None):
    """

    :param source_index: index of source node (of edge) or source graph (of node)
    :param score: 1-D Array
    :param k: Keep top k targets for each source
    :param ratio: Keep num_targets * ratio targets for each source
    :return: sampled_edge_index, sampled_edge_score, sample_index
    """

    if k is None and ratio is None:
        raise Exception("you should provide either k or ratio for topk_pool")
    elif k is not None and ratio is not None:
        raise Exception("you should provide either k or ratio for topk_pool, not both of them")

    # currently, we consider the source_index is not sorted
    # the option is preserved for future performance optimization
    source_index_sorted = False

    if source_index_sorted:
        sorted_source_index = source_index
        # sort score by source_index
        sorted_score = score
    else:
        source_index_perm = tf.argsort(source_index)
        sorted_source_index = tf.gather(source_index, source_index_perm)
        sorted_score = tf.gather(score, source_index_perm)

    sorted_score = tf.reshape(sorted_score, [-1])

    num_targets = tf.shape(sorted_source_index)[0]
    target_ones = tf.ones([num_targets], dtype=tf.int32)
    num_targets_for_sources = tf.math.segment_sum(target_ones, sorted_source_index)
    # number of columns for score matrix
    num_cols = tf.reduce_max(num_targets_for_sources)

    # max index of source + 1
    num_seen_sources = tf.shape(num_targets_for_sources)[0]

    min_score = tf.reduce_min(sorted_score)

    num_targets_before = tf.concat([
        tf.zeros([1], dtype=tf.int32),
        tf.math.cumsum(num_targets_for_sources)[:-1]
    ], axis=0)

    target_index_for_source = tf.range(0, num_targets) - tf.gather(num_targets_before, sorted_source_index)

    score_matrix = tf.cast(tf.fill([num_seen_sources, num_cols], min_score - 1.0), dtype=tf.float32)
    score_index = tf.stack([sorted_source_index, target_index_for_source], axis=1)
    score_matrix = tf.tensor_scatter_nd_update(score_matrix, score_index, sorted_score)

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

    row, col = tf.meshgrid(tf.range(num_seen_sources), tf.range(tf.reduce_max(node_k)), indexing="ij")
    row = tf.reshape(row, [-1])
    col = tf.reshape(col, [-1])
    repeated_k = tf.gather(node_k, row)
    k_mask = tf.less(col, repeated_k)

    row = tf.boolean_mask(row, k_mask)
    col = tf.boolean_mask(col, k_mask)

    sample_col_index = tf.gather_nd(sort_index, tf.stack([row, col], axis=1))

    topk_index = tf.gather(num_targets_before, row) + sample_col_index

    if source_index_sorted:
        return topk_index
    else:
        return tf.gather(source_index_perm, topk_index)
