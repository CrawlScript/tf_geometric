# coding=utf-8
import tensorflow as tf

from tf_geometric.utils.graph_utils import add_self_loop_edge
from tf_geometric.nn.kernel.segment import segment_count, segment_op_with_pad
from tf_geometric.nn.conv.gcn import gcn_mapper
from tf_geometric.nn.kernel.map_reduce import sum_reducer



# def sum_reducer(neighbor_msg, node_index, num_nodes=None):
#     return tf.math.unsorted_segment_sum(neighbor_msg, node_index, num_segments=num_nodes)

def mean_reducer(neighbor_msg, node_index, num_nodes=None):
    return tf.math.unsorted_segment_mean(neighbor_msg, node_index, num_segments=num_nodes)

def max_reducer(neighbor_msg, node_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = tf.reduce_max(node_index) + 1
    # max_x = tf.math.unsorted_segment_max(x, node_graph_index, num_segments=num_graphs)
    max_x = segment_op_with_pad(tf.math.segment_max, neighbor_msg, node_index, num_segments=num_nodes)
    return max_x


def mean_aggregate_neighbors(x, edge_index, kernel, edge_weight=None, bias=None,
                            activation=None, normalize=False):
    row, col = edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)


    if edge_weight is not None:
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

def mean_pooling_aggregate_neighbors(x, edge_index, kernel_1, kernel_2, edge_weight=None, bias_1=None,
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

    h = tf.concat([reduced_h, x],axis=1) @ kernel_2
    if bias_2 is not None:
        h += bias_2

    if activation is not None:
        h = activation(h)

    if normalize:
        h = tf.nn.l2_normalize(h,axis=-1)

    return h


def max_pooling_aggregate_neighbors(x, edge_index, kernel_1, kernel_2, edge_weight=None, bias_1=None,
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

    h = tf.concat([reduced_h, x], axis=1) @ kernel_2
    if bias_2 is not None:
        h += bias_2
    if activation is not None:
        h = activation(h)

    if normalize:
        h = tf.nn.l2_normalize(h, axis=-1)

    return h

# def lstm_aggregate_neighbors(x, edge_index, kernel_1, kernel_2=None, edge_weight=None, bias_1=None, bias_2=None,
#                 activation=None, normalize=False):
#
#     lstm = kernel_1
#
#     row, col = edge_index
#
#     nodes = tf.unique(row)
#
#     row = row.numpy()
#     nodes_ = nodes[0].numpy()
#
#     neighbor_feature = []
#     for i, v in enumerate(nodes_):
#         index = np.where(row == i)
#
#         neighbor_x = tf.gather(x, edge_index[1][index[0][0]:index[0][-1]+1])
#         # paddings = [[0, size - len(index[0])], [0, 0]]
#         # neighbor_x = tf.pad(neighbor_x, paddings,"CONSTANT")
#
#         neighbor_x = tf.expand_dims(neighbor_x, axis=0)
#
#         h = lstm(neighbor_x)
#         neighbor_feature.append(h)
#
#     h = np.array(neighbor_feature).squeeze(axis=1)
#
#     # h = lstm(neighbor_x)
#
#     # repeated_x = tf.gather(x, row)
#     # if edge_weight is not None:
#     #     h = gcn_mapper(repeated_x, h, edge_weight=edge_weight)
#
#     h = tf.concat([h, x], axis=1) @ kernel_2
#
#     if bias_1 is not None:
#         h += bias_2
#
#     if activation is not None:
#         h = activation(h)
#
#     if normalize:
#         h = tf.nn.l2_normalize(h, axis=-1)
#
#     return h


def graphSAGE(x, edge_index, edge_weight, kernel_1, kernel_2=None, bias_1=None, bias_2=None,
              aggregate_type='mean', activation=None, normalize=False):
    """

       :param x: Tensor, shape: [num_nodes, num_features], node features
       :param edge_index: Tensor, shape: [2, num_edges], edge information
       :param edge_weight: Tensor or None, shape: [num_edges]
       :param kernel_1: Tensor, shape: [num_features*2, num_output_features], weight
       :param bias_1: Tensor, shape: [num_output_features], bias
       :param kernel_2: Tensor, shape: [num_features+num_hidden_feature, num_output_features], weight
       :param bias_2: Tensor, shape: [num_output_features], bias
       :param activation: Activation function to use.
       :param normalize: If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
       :return: Updated node features (x), shape: [num_nodes, num_output_features]
       """

    if aggregate_type == 'mean':

        h = mean_aggregate_neighbors(x, edge_index, kernel_1, edge_weight=edge_weight, bias=bias_1,
                            activation=activation, normalize=normalize)

    if aggregate_type == 'gcn':

        h = gcn_aggregate_neighbors(x, edge_index, kernel_1, edge_weight=edge_weight, bias=bias_1,
                            activation=activation, normalize=normalize)


    if aggregate_type == 'mean_pooling':

        h = mean_pooling_aggregate_neighbors(x, edge_index, kernel_1, kernel_2, edge_weight=edge_weight,
                                             bias_1=bias_1, bias_2=bias_2, activation=activation, normalize=normalize)

    if aggregate_type == 'max_pooling':

        h = max_pooling_aggregate_neighbors(x, edge_index, kernel_1, kernel_2, edge_weight=edge_weight,
                                             bias_1=bias_1, bias_2=bias_2, activation=activation, normalize=normalize)



    return h




