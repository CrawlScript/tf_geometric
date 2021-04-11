# coding=utf-8
import tensorflow as tf
import numpy as np

from tf_geometric.nn import mean_reducer, max_reducer, sum_reducer
from tf_geometric.nn.conv.gcn import gcn_mapper, gcn_norm_edge


def mean_graph_sage(x, edge_index, edge_weight,
                    self_kernel,
                    neighbor_kernel,
                    bias=None,
                    activation=None,
                    concat=True, normalize=False):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param self_kernel: Tensor, shape: [num_features, num_hidden_units], weight
    :param neighbor_kernel: Tensor, shape: [num_features, num_hidden_units], weight.
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param normalize: If set to :obj:`True`, output features
                will be :math:`\ell_2`-normalized, *i.e.*,
                :math:`\frac{\mathbf{x}^{\prime}_i}
                {\| \mathbf{x}^{\prime}_i \|_2}`.
                (default: :obj:`False`)
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """
    num_nodes = tf.shape(x)[0]
    # num_edges = tf.shape(edge_index)[1]

    row, col = edge_index[0], edge_index[1]
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    if edge_weight is not None:
        neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

    neighbor_reduced_msg = mean_reducer(neighbor_x, row, num_nodes=num_nodes)

    neighbor_msg = neighbor_reduced_msg @ neighbor_kernel
    x = x @ self_kernel

    if concat:
        h = tf.concat([x, neighbor_msg], axis=1)
    else:
        h = x + neighbor_msg

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    if normalize:
        h = tf.nn.l2_normalize(h, axis=-1)

    return h



def sum_graph_sage(x, edge_index, edge_weight,
                    self_kernel,
                    neighbor_kernel,
                    bias=None,
                    activation=None,
                    concat=True, normalize=False):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param self_kernel: Tensor, shape: [num_features, num_hidden_units], weight
    :param neighbor_kernel: Tensor, shape: [num_features, num_hidden_units], weight.
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param normalize: If set to :obj:`True`, output features
                will be :math:`\ell_2`-normalized, *i.e.*,
                :math:`\frac{\mathbf{x}^{\prime}_i}
                {\| \mathbf{x}^{\prime}_i \|_2}`.
                (default: :obj:`False`)
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """
    num_nodes = tf.shape(x)[0]
    # num_edges = tf.shape(edge_index)[1]

    row, col = edge_index[0], edge_index[1]
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    if edge_weight is not None:
        neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

    neighbor_reduced_msg = sum_reducer(neighbor_x, row, num_nodes=num_nodes)

    neighbor_msg = neighbor_reduced_msg @ neighbor_kernel
    x = x @ self_kernel

    if concat:
        h = tf.concat([x, neighbor_msg], axis=1)
    else:
        h = x + neighbor_msg

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    if normalize:
        h = tf.nn.l2_normalize(h, axis=-1)

    return h


def gcn_graph_sage(x, edge_index, edge_weight, kernel, bias=None, activation=None,
                   normalize=False, cache=None):
    """

        :param x: Tensor, shape: [num_nodes, num_features], node features
        :param edge_index: Tensor, shape: [2, num_edges], edge information
        :param edge_weight: Tensor or None, shape: [num_edges]
        :param kernel: Tensor, shape: [num_features, num_output_features], weight
        :param bias: Tensor, shape: [num_output_features], bias
        :param activation: Activation function to use.
        :param normalize: If set to :obj:`True`, output features
                will be :math:`\ell_2`-normalized, *i.e.*,
                :math:`\frac{\mathbf{x}^{\prime}_i}
                {\| \mathbf{x}^{\prime}_i \|_2}`.
                (default: :obj:`False`)
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """
    num_nodes = tf.shape(x)[0]
    num_edges = tf.shape(edge_index)[1]

    if edge_weight is not None:
        edge_weight = tf.ones([num_edges], dtype=tf.float32)

    updated_edge_index, normed_edge_weight = gcn_norm_edge(edge_index, num_nodes, edge_weight, cache)
    row, col = updated_edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=normed_edge_weight)

    reduced_msg = sum_reducer(neighbor_x, row, num_nodes=num_nodes)

    h = reduced_msg @ kernel
    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    if normalize:
        h = tf.nn.l2_normalize(h, axis=-1)

    return h


def mean_pool_graph_sage(x, edge_index, edge_weight,
                         self_kernel, neighbor_mlp_kernel, neighbor_kernel,
                         neighbor_mlp_bias=None, bias=None, activation=None,
                         concat=True, normalize=False):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param self_kernel: Tensor, shape: [num_features, num_hidden_units], weight.
    :param neighbor_mlp_kernel: Tensor, shape: [num_features, num_hidden_units]. weight.
    :param neighbor_kernel: Tensor, shape: [num_hidden_units, num_hidden_units], weight.
    :param neighbor_mlp_bias: Tensor, shape: [num_hidden_units * 2], bias
    :param bias: Tensor, shape: [num_output_features], bias.
    :param activation: Activation function to use.
    :param normalize: If set to :obj:`True`, output features
                will be :math:`\ell_2`-normalized, *i.e.*,
                :math:`\frac{\mathbf{x}^{\prime}_i}
                {\| \mathbf{x}^{\prime}_i \|_2}`.
                (default: :obj:`False`)
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    num_nodes = tf.shape(x)[0]
    num_edges = tf.shape(edge_index)[1]

    if edge_weight is not None:
        edge_weight = tf.ones([num_edges], dtype=tf.float32)

    row, col = edge_index[0], edge_index[1]
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

    h = neighbor_x @ neighbor_mlp_kernel
    if neighbor_mlp_bias is not None:
        h += neighbor_mlp_bias

    if activation is not None:
        h = activation(h)

    reduced_h = mean_reducer(h, row, num_nodes=num_nodes)

    from_neighbor = reduced_h @ neighbor_kernel
    from_x = x @ self_kernel

    if concat:
        output = tf.concat([from_x, from_neighbor], axis=1)
    else:
        output = from_x + from_neighbor

    if bias is not None:
        output += bias

    if activation is not None:
        output = activation(output)

    if normalize:
        output = tf.nn.l2_normalize(output, axis=-1)

    return output


def max_pool_graph_sage(x, edge_index, edge_weight,
                        self_kernel, neighbor_mlp_kernel, neighbor_kernel,
                        neighbor_mlp_bias=None, bias=None, activation=None,
                        concat=True, normalize=False):
    """

            :param x: Tensor, shape: [num_nodes, num_features], node features
            :param edge_index: Tensor, shape: [2, num_edges], edge information
            :param edge_weight: Tensor or None, shape: [num_edges]
            :param self_kernel: Tensor, shape: [num_features, num_hidden_units], weight.
            :param neighbor_mlp_kernel: Tensor, shape: [num_features, num_hidden_units]. weight.
            :param neighbor_kernel: Tensor, shape: [num_hidden_units, num_hidden_units], weight.
            :param neighbor_mlp_bias: Tensor, shape: [num_hidden_units * 2], bias
            :param bias: Tensor, shape: [num_output_features], bias.
            :param activation: Activation function to use.
            :param normalize: If set to :obj:`True`, output features
                        will be :math:`\ell_2`-normalized, *i.e.*,
                        :math:`\frac{\mathbf{x}^{\prime}_i}
                        {\| \mathbf{x}^{\prime}_i \|_2}`.
                        (default: :obj:`False`)
            :return: Updated node features (x), shape: [num_nodes, num_output_features]
            """
    num_nodes = tf.shape(x)[0]
    num_edges = tf.shape(edge_index)[1]

    if edge_weight is not None:
        edge_weight = tf.ones([num_edges], dtype=tf.float32)

    row, col = edge_index[0], edge_index[1]
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

    h = neighbor_x @ neighbor_mlp_kernel
    if neighbor_mlp_bias is not None:
        h += neighbor_mlp_bias

    if activation is not None:
        h = activation(h)

    reduced_h = max_reducer(h, row, num_nodes=num_nodes)
    from_neighs = reduced_h @ neighbor_kernel
    from_x = x @ self_kernel

    if concat:
        output = tf.concat([from_x, from_neighs], axis=1)
    else:
        output = from_x + from_neighs

    if bias is not None:
        output += bias

    if activation is not None:
        output = activation(output)

    if normalize:
        output = tf.nn.l2_normalize(output, axis=-1)

    return output


def lstm_graph_sage(x, edge_index, lstm, self_kernel, neighbor_kernel,
                    bias=None, activation=None,
                    concat=True, normalize=False, training=False):
    """


    :param x: Tensor, shape: [num_nodes, num_features], node features.
    :param edge_index: Tensor, shape: [2, num_edges], edge information.
    :param lstm: Long Short-Term Merory.
    :param self_kernel: Tensor, shape: [num_features, num_hidden_units], weight.
    :param neighbor_kernel: Tensor, shape: [num_hidden_units, num_hidden_units], weight.
    :param bias: Tensor, shape: [num_output_features], bias.
    :param activation: Activation function to use.
    :param normalize: If set to :obj:`True`, output features
                will be :math:`\ell_2`-normalized, *i.e.*,
                :math:`\frac{\mathbf{x}^{\prime}_i}
                {\| \mathbf{x}^{\prime}_i \|_2}`.
                (default: :obj:`False`)
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """
    num_nodes = tf.shape(x)[0]
    num_edges = tf.shape(edge_index)[1]

    row, col = edge_index[0], edge_index[1]

    sort_index = tf.argsort(row, axis=-1, direction='ASCENDING')
    row = tf.gather(row, sort_index)
    col = tf.gather(col, sort_index)

    degree = tf.math.unsorted_segment_sum(tf.ones([num_edges], dtype=tf.int32), row, num_nodes)

    num_sampled_neighbors = tf.reduce_max(degree)
    num_edges_before_row = tf.concat([tf.zeros([1], dtype=tf.int32), tf.math.cumsum(degree)[:-1]], axis=0)
    matrix_col_index = tf.range(num_edges) - tf.gather(num_edges_before_row, row)

    neighbor_matrix = tf.fill((num_nodes, num_sampled_neighbors), num_nodes)
    neighbor_matrix = tf.tensor_scatter_nd_update(
        neighbor_matrix,
        tf.stack([row, matrix_col_index], axis=1),
        col
    )

    padded_x = tf.concat([x, tf.zeros([1, tf.shape(x)[-1]], x.dtype)], axis=0)

    neighbor_x = tf.gather(padded_x, neighbor_matrix)
    neighbor_h = lstm(neighbor_x, training=training)

    reduced_h = tf.reduce_mean(neighbor_h, axis=1)

    from_neighbor = reduced_h @ neighbor_kernel
    from_x = x @ self_kernel

    if concat:
        output = tf.concat([from_x, from_neighbor], axis=1)
    else:
        output = from_x + from_neighbor

    if bias is not None:
        output += bias

    if activation is not None:
        output = activation(output)

    if normalize:
        output = tf.nn.l2_normalize(output, axis=-1)

    return output
