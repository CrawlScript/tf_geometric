import tensorflow as tf
import numpy as np
from tf_geometric.nn.kernel.segment import segment_count, segment_op_with_pad
from tf_geometric.nn.conv.gcn import gcn_mapper
from tf_geometric.utils.graph_utils import add_self_loop_edge
from collections import Counter


def mean_reducer(neighbor_msg, node_index, num_nodes=None):
    return tf.math.unsorted_segment_mean(neighbor_msg, node_index, num_segments=num_nodes)


def sum_reducer(neighbor_msg, node_index, num_nodes=None):
    return tf.math.unsorted_segment_sum(neighbor_msg, node_index, num_segments=num_nodes)


def max_reducer(neighbor_msg, node_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = tf.reduce_max(node_index) + 1
    # max_x = tf.math.unsorted_segment_max(x, node_graph_index, num_segments=num_graphs)
    max_x = segment_op_with_pad(tf.math.segment_max, neighbor_msg, node_index, num_segments=num_nodes)
    return max_x


def gcn_norm_edge(edge_index, num_nodes, edge_weight=None, renorm=True, improved=False, cache=None):
    cache_key = "gcn_normed_edge"

    if cache is not None and cache_key in cache and cache[cache_key] is not None:
        return cache[cache_key]

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    fill_weight = 2.0 if improved else 1.0

    if renorm:
        edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight=edge_weight,
                                                     fill_weight=fill_weight)

    row, col = edge_index
    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=num_nodes)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    normed_edge_weight = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

    if not renorm:
        edge_index, normed_edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight=normed_edge_weight,
                                                            fill_weight=fill_weight)

    if cache is not None:
        cache[cache_key] = edge_index, normed_edge_weight

    return edge_index, normed_edge_weight


def mean_graph_sage(x, edge_index, edge_weight, neighs_kernel, self_kernel, bias=None, activation=None,
                    normalize=False):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param neighs_kernel: Tensor, shape: [num_features, num_hidden_units], weight.
    :param self_kernel: Tensor, shape: [num_features, num_hidden_units], weight
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param normalize: If set to :obj:`True`, output features
                will be :math:`\ell_2`-normalized, *i.e.*,
                :math:`\frac{\mathbf{x}^{\prime}_i}
                {\| \mathbf{x}^{\prime}_i \|_2}`.
                (default: :obj:`False`)
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    row, col = edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    if edge_weight is not None:
        neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

    neighbor_reduced_msg = mean_reducer(neighbor_x, row, num_nodes=len(x))

    neighbor_msg = neighbor_reduced_msg @ neighs_kernel
    x = x @ self_kernel
    h = tf.concat([neighbor_msg, x], axis=1)

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
    if edge_weight is not None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    updated_edge_index, normed_edge_weight = gcn_norm_edge(edge_index, x.shape[0], edge_weight, cache)
    row, col = updated_edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=normed_edge_weight)

    reduced_msg = sum_reducer(neighbor_x, row, num_nodes=x.shape[0])

    h = reduced_msg @ kernel
    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    if normalize:
        h = tf.nn.l2_normalize(h, axis=-1)

    return h


def mean_pooling_graph_sage(x, edge_index, edge_weight, mlp_kernel, neighs_kernel, self_kernel, dropout,
                            mlp_bias=None, bias=None, activation=None, normalize=False):
    """

        :param x: Tensor, shape: [num_nodes, num_features], node features
        :param edge_index: Tensor, shape: [2, num_edges], edge information
        :param edge_weight: Tensor or None, shape: [num_edges]
        :param mlp_kernel: Tensor, shape: [num_features, num_hidden_units]. weight.
        :param neighs_kernel: Tensor, shape: [num_hidden_units, num_hidden_units], weight.
        :param self_kernel: Tensor, shape: [num_features, num_hidden_units], weight.
        :param dropout: Dropout function.
        :param mlp_bias: Tensor, shape: [num_hidden_units * 2], bias
        :param bias: Tensor, shape: [num_output_features], bias.
        :param activation: Activation function to use.
        :param normalize: If set to :obj:`True`, output features
                    will be :math:`\ell_2`-normalized, *i.e.*,
                    :math:`\frac{\mathbf{x}^{\prime}_i}
                    {\| \mathbf{x}^{\prime}_i \|_2}`.
                    (default: :obj:`False`)
        :return: Updated node features (x), shape: [num_nodes, num_output_features]
        """

    if edge_weight is not None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    row, col = edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

    neighbor_x = dropout(neighbor_x)
    h = neighbor_x @ mlp_kernel
    if mlp_bias is not None:
        h += mlp_bias

    if activation is not None:
        h = activation(h)

    reduced_h = mean_reducer(h, row, num_nodes=len(x))

    from_neighs = reduced_h @ neighs_kernel
    from_x = x @ self_kernel

    output = tf.concat([from_neighs, from_x], axis=1)
    if bias is not None:
        output += bias

    if activation is not None:
        output = activation(output)

    if normalize:
        output = tf.nn.l2_normalize(output, axis=-1)

    return output


def max_pooling_graph_sage(x, edge_index, edge_weight, mlp_kernel, neighs_kernel, self_kernel, dropout,
                           mlp_bias=None, bias=None, activation=None, normalize=False):
    """

            :param x: Tensor, shape: [num_nodes, num_features], node features
            :param edge_index: Tensor, shape: [2, num_edges], edge information
            :param edge_weight: Tensor or None, shape: [num_edges]
            :param mlp_kernel: Tensor, shape: [num_features, num_hidden_units]. weight.
            :param neighs_kernel: Tensor, shape: [num_hidden_units, num_hidden_units], weight.
            :param self_kernel: Tensor, shape: [num_features, num_hidden_units], weight.
            :param dropout: Dropout function.
            :param mlp_bias: Tensor, shape: [num_hidden_units * 2], bias
            :param bias: Tensor, shape: [num_output_features], bias.
            :param activation: Activation function to use.
            :param normalize: If set to :obj:`True`, output features
                        will be :math:`\ell_2`-normalized, *i.e.*,
                        :math:`\frac{\mathbf{x}^{\prime}_i}
                        {\| \mathbf{x}^{\prime}_i \|_2}`.
                        (default: :obj:`False`)
            :return: Updated node features (x), shape: [num_nodes, num_output_features]
            """
    if edge_weight is not None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    row, col = edge_index
    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

    neighbor_x = dropout(neighbor_x)
    h = neighbor_x @ mlp_kernel
    if mlp_bias is not None:
        h += mlp_bias

    if activation is not None:
        h = activation(h)

    reduced_h = max_reducer(h, row, num_nodes=len(x))

    from_neighs = reduced_h @ neighs_kernel
    from_x = x @ self_kernel

    output = tf.concat([from_neighs, from_x], axis=1)
    if bias is not None:
        output += bias

    if activation is not None:
        output = activation(output)

    if normalize:
        output = tf.nn.l2_normalize(output, axis=-1)

    return output


def lstm_graph_sage(x, edge_index, edge_weight, lstm, neighs_kernel, self_kernel,
                    bias=None, activation=None, normalize=False):
    """

            :param x: Tensor, shape: [num_nodes, num_features], node features.
            :param edge_index: Tensor, shape: [2, num_edges], edge information.
            :param edge_weight: Tensor or None, shape: [num_edges].
            :param lstm: Long Short-Term Merory.
            :param neighs_kernel: Tensor, shape: [num_hidden_units, num_hidden_units], weight.
            :param self_kernel: Tensor, shape: [num_features, num_hidden_units], weight.
            :param mlp_bias: Tensor, shape: [num_hidden_units * 2], bias
            :param bias: Tensor, shape: [num_output_features], bias.
            :param activation: Activation function to use.
            :param normalize: If set to :obj:`True`, output features
                        will be :math:`\ell_2`-normalized, *i.e.*,
                        :math:`\frac{\mathbf{x}^{\prime}_i}
                        {\| \mathbf{x}^{\prime}_i \|_2}`.
                        (default: :obj:`False`)
            :return: Updated node features (x), shape: [num_nodes, num_output_features]
            """

    if edge_weight is not None:
        edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

    row, col = edge_index
    # repeated_x = tf.gather(x, row)
    # neighbor_x = tf.gather(x, col)
    # neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)
    padding_feature = tf.zeros((1, x.shape[-1]), dtype=tf.float32)
    x_ = tf.concat((x, padding_feature), axis=0)

    num_neighs = Counter()

    col_numpy = col.numpy()
    row_numpy = row.numpy()
    for i in row_numpy:
        num_neighs[i] += 1

    max_len = max(num_neighs.values())
    neighbors = np.zeros((x.shape[0], max_len), dtype=np.int)
    nums = 0
    for i, v in num_neighs.items():
        neighbors[i][0:v] = col_numpy[nums:nums + v]
        nums += v
        neighbors[i][v:max_len] = x.shape[0]

    neighbor_feature = tf.gather(x_, neighbors)

    reduced_h = lstm(neighbor_feature)
    a = list(num_neighs.values())

    reduced_h = [tf.reduce_mean(h[:a[i]], axis=0) for i,h in enumerate(reduced_h)]
    reduced_h = tf.reshape(reduced_h,[x.shape[0],-1])

    from_x = x @ self_kernel

    output = tf.concat([reduced_h, from_x], axis=1)
    if bias is not None:
        output += bias

    if activation is not None:
        output = activation(output)

    if normalize:
        output = tf.nn.l2_normalize(output, axis=-1)

    return output
