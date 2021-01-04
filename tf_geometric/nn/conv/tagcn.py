# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_updater
from tf_geometric.nn.conv.gcn import gcn_norm_edge, gcn_mapper


def tagcn(x, edge_index, edge_weight, K, kernel, bias=None, activation=None, renorm=False, improved=False, cache=None):
    """
    Functional API for Topology Adaptive Graph Convolutional Network (TAGCN).

    :param x: Tensor, shape: [num_nodes, num_features], node features.
    :param edge_index: Tensor, shape: [2, num_edges], edge information.
    :param edge_weight: Tensor or None, shape: [num_edges].
    :param K: Number of hops.(default: :obj:`3`)
    :param kernel: Tensor, shape: [num_features, num_output_features], weight.
    :param bias: Tensor, shape: [num_output_features], bias.
    :param activation: Activation function to use.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    xs = [x]
    updated_edge_index, normed_edge_weight = gcn_norm_edge(edge_index, x.shape[0], edge_weight,
                                                           renorm, improved, cache)
    for k in range(K):
        h = aggregate_neighbors(
            xs[-1], updated_edge_index, normed_edge_weight,
            gcn_mapper,
            sum_reducer,
            identity_updater
        )

        xs.append(h)

    h = tf.concat(xs, axis=-1)

    out = h @ kernel
    if bias is not None:
        out += bias

    if activation is not None:
        out = activation(out)

    return out
