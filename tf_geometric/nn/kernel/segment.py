# coding=utf-8
import tensorflow as tf


def segment_op_with_pad(segment_op, data, segment_ids, num_segments):

    sort_index = tf.argsort(segment_ids)
    sorted_segment_ids = tf.gather(segment_ids, sort_index)
    sorted_data = tf.gather(data, sort_index)

    reduced_data = segment_op(sorted_data, sorted_segment_ids)
    num_paddings = num_segments - tf.shape(reduced_data)[0]

    pad_shape = tf.concat([
        [num_paddings],
        tf.shape(data)[1:]
    ], axis=0)
    pads = tf.zeros(pad_shape, dtype=reduced_data.dtype)
    outputs = tf.concat(
        [reduced_data, pads],
        axis=0
    )
    return outputs


def segment_softmax(data, segment_ids, num_segments):
    max_values = tf.math.unsorted_segment_max(data, segment_ids, num_segments=num_segments)
    gathered_max_values = tf.gather(max_values, segment_ids)
    exp = tf.exp(data - tf.stop_gradient(gathered_max_values))
    denominator = tf.math.unsorted_segment_sum(exp, segment_ids, num_segments=num_segments) + 1e-8
    gathered_denominator = tf.gather(denominator, segment_ids)
    score = exp / gathered_denominator
    return score


def segment_count(index, num_segments=None):
    data = tf.ones_like(index)
    if num_segments is None:
        num_segments = tf.reduce_max(index) + 1
    return tf.math.unsorted_segment_sum(data, index, num_segments=num_segments)
