# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.kernel.segment import segment_softmax

from tf_geometric.utils.union_utils import union_len


def set2set(x, node_graph_index, lstm, num_iterations, training=None):
    """
    Functional API for Set2Set

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param node_graph_index: Tensor/NDArray, shape: [num_nodes], graph index for each node
    :param lstm: A lstm model.
    :param num_iterations: Number of iterations for attention.
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: Graph features, shape: [num_graphs, num_node_features * 2]
    """

    num_graphs = tf.reduce_max(node_graph_index) + 1

    lstm_units = tf.shape(x)[-1]

    h = tf.zeros([num_graphs, lstm_units * 2], dtype=tf.float32)
    initial_state = [tf.zeros([1, lstm_units], dtype=tf.float32), tf.zeros([1, lstm_units], dtype=tf.float32)]

    for _ in range(num_iterations):

        h = tf.expand_dims(h, axis=0)
        h, state_h, state_c = lstm(h, initial_state=initial_state, training=training)
        initial_state = [state_h, state_c]
        h = tf.squeeze(h, axis=0)

        repeated_h = tf.gather(h, node_graph_index)
        # attention
        att_score = tf.reduce_sum(x * repeated_h, axis=-1, keepdims=True)
        normed_att_score = segment_softmax(att_score, node_graph_index, num_graphs)
        att_h = tf.math.unsorted_segment_sum(x * normed_att_score, node_graph_index, num_graphs)
        h = tf.concat([h, att_h], axis=-1)

    return h
