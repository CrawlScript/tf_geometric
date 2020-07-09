# coding=utf-8
import tensorflow as tf

from tf_geometric.utils.graph_utils import add_self_loop_edge
from tf_geometric.nn.kernel.segment import segment_count, segment_op_with_pad
from tf_geometric.nn.conv.gcn import gcn_mapper
from tf_geometric.nn.kernel.map_reduce import sum_reducer




def mean_reducer(neighbor_msg, node_index, num_nodes=None):
    return tf.math.unsorted_segment_mean(neighbor_msg, node_index, num_segments=num_nodes)

def max_reducer(neighbor_msg, node_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = tf.reduce_max(node_index) + 1
    # max_x = tf.math.unsorted_segment_max(x, node_graph_index, num_segments=num_graphs)
    max_x = segment_op_with_pad(tf.math.segment_max, neighbor_msg, node_index, num_segments=num_nodes)
    return max_x



def mean_aggregate_neighbors(x, edge_index, kernel_1, kernel_2, edge_weight=None, bias=None,
                             activation=None, normalize=False):
    row, col = edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    if edge_weight is not None:
        neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)


    neighbor_reduced_msg = mean_reducer(neighbor_x, row, num_nodes=len(x))

    neighbor_msg = neighbor_reduced_msg @ kernel_1
    x = x @ kernel_2
    h = tf.concat([neighbor_msg, x], axis=1)

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    if normalize:
        h = tf.nn.l2_normalize(h,axis=-1)

    return h

def gcn_aggregate_neighbors(x, edge_index, kernel, edge_weight=None, bias=None,
                             activation=None, normalize=False):

    if edge_weight is not None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    edge_index, edge_weight = add_self_loop_edge(edge_index, x.shape[0], edge_weight=edge_weight, fill_weight=2.0)

    row, col = edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

    reduced_msg = mean_reducer(neighbor_x, row, num_nodes=len(x))
    updated_msg = tf.concat([reduced_msg, x], axis=1)


    h = updated_msg @ kernel
    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    if normalize:
        h = tf.nn.l2_normalize(h,axis=-1)

    return h

def mean_pooling_aggregate_neighbors(x, edge_index, kernel_1, kernel_2, kernel_3, edge_weight=None, bias_1=None,
                            bias_2=None, activation=None, normalize=False):

    if edge_weight is not None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    edge_index, edge_weight = add_self_loop_edge(edge_index, x.shape[0], edge_weight=edge_weight, fill_weight=2.0)

    row, col = edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

    h = neighbor_x @ kernel_1
    if bias_1 is not None:
        h += bias_1

    if activation is not None:
        h = activation(h)

    reduced_h = mean_reducer(h, row, num_nodes=len(x))

    from_neighs = reduced_h @ kernel_2
    from_x = x @ kernel_3

    output = tf.concat([from_neighs, from_x],axis=1)
    if bias_2 is not None:
        output += bias_2

    if activation is not None:
        output = activation(output)

    if normalize:
        output = tf.nn.l2_normalize(output,axis=-1)

    return output


def max_pooling_aggregate_neighbors(x, edge_index, kernel_1, kernel_2, kernel_3, edge_weight=None, bias_1=None,
                            bias_2=None, activation=None, normalize=False):

    if edge_weight is not None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    # edge_index, edge_weight = add_self_loop_edge(edge_index, x.shape[0], edge_weight=edge_weight, fill_weight=2.0)

    row, col = edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

    h = neighbor_x @ kernel_1
    if bias_1 is not None:
        h += bias_1

    if activation is not None:
        h = activation(h)

    reduced_h = max_reducer(h, row, num_nodes=len(x))

    from_neighs = reduced_h @ kernel_2
    from_x = x @ kernel_3

    output = tf.concat([from_neighs, from_x],axis=1)
    if bias_2 is not None:
        output += bias_2

    if activation is not None:
        output = activation(output)

    if normalize:
        output = tf.nn.l2_normalize(output,axis=-1)

    return output



def graphSAGE(x, edge_index, edge_weight, kernel_1, kernel_2=None, kernel_3=None, bias_1=None, bias_2=None,
              aggregate_type='mean', activation=None, normalize=False):
    """

           :param x: Tensor, shape: [num_nodes, num_features], node features
           :param edge_index: Tensor, shape: [2, num_edges], edge information
           :param edge_weight: Tensor or None, shape: [num_edges]
           :param kernel_1: Tensor, in "mean_pooling" or "max_pooling": shape: [num_features, hidden_dim], weight;
           in "mean", shape: [num_features, hidden_dim], weight; in "gcn", shape[num_features*2, hidden_dim], weight;
           :param kernel_2: Tensor, in "mean_pooling" or "max_pooling": shape: [hidden_dim, hidden_dim], weight;
           in "mean", shape: [num_features, hidden_dim], weight; in "gcn", the kernel is none.
           :param kernel_3: Tensor, in "mean_pooling" or "max_pooling": shape: [num_features, hidden_dim], weight;
           in "mean",the kernel is none; in "gcn",the kernel is none.
           :param bias_1: Tensor, in "mean_pooling" or "max_pooling" or "gcn", shape: [hidden_dim], bias; in "mean", shape: [hidden_dim * 2], bias.
           :param bias_2: Tensor, in "mean_pooling" or "max_pooling", shape: [hidden_dim * 2], bias; in "gcn" and "mean", none.
           :param activation: Activation function to use.
           :param normalize: If set to :obj:`True`, output features
                will be :math:`\ell_2`-normalized, *i.e.*,
                :math:`\frac{\mathbf{x}^{\prime}_i}
                {\| \mathbf{x}^{\prime}_i \|_2}`.
                (default: :obj:`False`)
           :return: Updated node features (x), shape: [num_nodes, num_output_features]
           """

    if aggregate_type == 'mean':

        h = mean_aggregate_neighbors(x, edge_index, kernel_1, kernel_2,  edge_weight=edge_weight, bias=bias_1,
                                     activation=activation, normalize=normalize)

    if aggregate_type == 'gcn':

        h = gcn_aggregate_neighbors(x, edge_index, kernel_1, edge_weight=edge_weight, bias=bias_1,
                            activation=activation, normalize=normalize)


    if aggregate_type == 'mean_pooling':

        h = mean_pooling_aggregate_neighbors(x, edge_index, kernel_1, kernel_2, kernel_3, edge_weight=edge_weight,
                                             bias_1=bias_1, bias_2=bias_2, activation=activation, normalize=normalize)

    if aggregate_type == 'max_pooling':

        h = max_pooling_aggregate_neighbors(x, edge_index, kernel_1, kernel_2, kernel_3, edge_weight=edge_weight,
                                             bias_1=bias_1, bias_2=bias_2, activation=activation, normalize=normalize)



    return h




