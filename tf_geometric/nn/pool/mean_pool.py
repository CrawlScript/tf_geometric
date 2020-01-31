# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.kernel.segment import segment_count


def mean_pool(x, node_graph_index, num_graphs=None):
    if num_graphs is None:
        num_graphs = tf.reduce_max(node_graph_index) + 1
    num_nodes_of_graphs = segment_count(node_graph_index, num_segments=num_graphs)
    sum_x = tf.math.unsorted_segment_sum(x, node_graph_index, num_segments=num_graphs)
    return sum_x / tf.cast(tf.expand_dims(num_nodes_of_graphs, -1), tf.float32)








