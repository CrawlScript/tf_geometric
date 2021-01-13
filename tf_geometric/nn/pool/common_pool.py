# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.kernel.segment import segment_count, segment_op_with_pad


def mean_pool(x, node_graph_index, num_graphs=None):
    if num_graphs is None:
        num_graphs = tf.reduce_max(node_graph_index) + 1
    num_nodes_of_graphs = segment_count(node_graph_index, num_segments=num_graphs)
    sum_x = tf.math.unsorted_segment_sum(x, node_graph_index, num_segments=num_graphs)
    return sum_x / (tf.cast(tf.expand_dims(num_nodes_of_graphs, -1), tf.float32) + 1e-8)


def sum_pool(x, node_graph_index, num_graphs=None):
    if num_graphs is None:
        num_graphs = tf.reduce_max(node_graph_index) + 1
    sum_x = tf.math.unsorted_segment_sum(x, node_graph_index, num_segments=num_graphs)
    return sum_x


if tf.__version__[0] == "1":

    def max_pool(x, node_graph_index, num_graphs=None):
        if num_graphs is None:
            num_graphs = tf.reduce_max(node_graph_index) + 1
        # max_x = tf.math.unsorted_segment_max(x, node_graph_index, num_segments=num_graphs)
        max_x = segment_op_with_pad(tf.math.segment_max, x, node_graph_index, num_segments=num_graphs)
        return max_x


    def min_pool(x, node_graph_index, num_graphs=None):
        if num_graphs is None:
            num_graphs = tf.reduce_max(node_graph_index) + 1
        # min_x = tf.math.unsorted_segment_min(x, node_graph_index, num_segments=num_graphs)
        min_x = segment_op_with_pad(tf.math.segment_min, x, node_graph_index, num_segments=num_graphs)
        return min_x

else:

    def max_pool(x, node_graph_index, num_graphs=None):
        if num_graphs is None:
            num_graphs = tf.reduce_max(node_graph_index) + 1
        max_x = tf.math.unsorted_segment_max(x, node_graph_index, num_segments=num_graphs)
        return max_x


    def min_pool(x, node_graph_index, num_graphs=None):
        if num_graphs is None:
            num_graphs = tf.reduce_max(node_graph_index) + 1
        min_x = tf.math.unsorted_segment_min(x, node_graph_index, num_segments=num_graphs)
        return min_x






