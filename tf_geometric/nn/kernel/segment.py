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
