# coding=utf-8
import tensorflow as tf
import numpy as np


def drop_edge(inputs, rate=0.5, force_undirected=False, training=None):
    """

    :param inputs: List of edge_index and other edge attributes [edge_index, edge_attr, ...]
    :param rate: dropout rate
    :param force_undirected: If set to `True`, will either
            drop or keep both edges of an undirected edge.
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: List of dropped edge_index and other dropped edge attributes
    """

    if not training:
        return inputs

    if rate < 0.0 or rate > 1.0:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(rate))

    edge_index, *edge_attrs = inputs

    edge_index_is_tensor = tf.is_tensor(edge_index)
    if not edge_index_is_tensor:
        edge_index = tf.convert_to_tensor(edge_index)

    row, col = edge_index[0], edge_index[1]
    if force_undirected:
        index = tf.where(tf.less(row, col))
        index = tf.boolean_mask(index, tf.greater(tf.nn.dropout(tf.ones_like(index, dtype=tf.float32), rate), 0))
        dropped_edge_index = tf.gather(edge_index, index, axis=-1)
        dropped_edge_index = tf.concat([dropped_edge_index, tf.gather(dropped_edge_index, [1, 0])], axis=-1)
        index = tf.concat([index, index], axis=-1)
    else:
        index = tf.boolean_mask(tf.range(0, tf.shape(row)[0]),
                                tf.greater(tf.nn.dropout(tf.ones_like(row, dtype=tf.float32), rate), 0))
        dropped_edge_index = tf.gather(edge_index, index, axis=-1)

    for i in range(len(edge_attrs)):
        if tf.is_tensor(edge_attrs[i]):
            edge_attrs[i] = tf.gather(edge_attrs[i], index, axis=-1)
        else:
            edge_attrs[i] = np.take(edge_attrs[i], index, axis=-1)

    if not edge_index_is_tensor:
        dropped_edge_index = dropped_edge_index.numpy()

    return [dropped_edge_index] + edge_attrs
