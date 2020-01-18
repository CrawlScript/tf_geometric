# coding=utf-8
import tensorflow as tf


# def segment_op_with_pad(segment_op, data, segment_ids, total):
#     reduced_data = segment_op(data, segment_ids)
#     num_paddings = total - len(reduced_data)
#
#     pads = tf.zeros([num_paddings] + data.shape.as_list()[1:], dtype=reduced_data.dtype)
#     outputs = tf.concat(
#         [reduced_data, pads],
#         axis=0
#     )
#     return outputs
#
#
# def segment_sum_with_pad(data, segment_ids, total):
#     return segment_op_with_pad(tf.math.segment_sum, data, segment_ids, total)

def segment_softmax(data, segment_ids, num_segments):
    exp = tf.exp(data)
    denominator = tf.math.unsorted_segment_sum(data, segment_ids, num_segments=num_segments)
    gathered_denominator = tf.gather(denominator, segment_ids)
    score = exp * tf.pow(gathered_denominator, -1)
    return score


def segment_count(index, num_segments=None):
    data = tf.ones_like(index)
    if num_segments is None:
        return tf.math.segment_sum(data, index)
    else:
        return tf.math.unsorted_segment_sum(data, index, num_segments=num_segments)
