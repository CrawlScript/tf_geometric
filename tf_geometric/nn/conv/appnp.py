# coding=utf-8

import tensorflow as tf

from tf_geometric.nn.conv.gcn import gcn_norm_edge, gcn_mapper
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_updater, sum_reducer, identity_updater


def appnp(x, edge_index, edge_weight, kernels, biases,
          dense_activation=tf.nn.relu, activation=None,
          num_iterations=2, alpha=0.15,
          dense_drop_rate=0.0, edge_drop_rate=0.0, cache=None, training=False):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        To use @tf_utils.function with gcn, you should cache the noremd edge information before the first call of the gcn.
        (1) If you're using OOP APIs tfg.layers.GCN:
            gcn_layer.cache_normed_edge(graph)
        (2) If you're using functional API tfg.nn.gcn:
            from tf_geometric.nn.conv.gcn import gcn_cache_normed_edge
            gcn_cache_normed_edge(graph)
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    num_nodes = tf.shape(x)[0]
    updated_edge_index, normed_edge_weight = gcn_norm_edge(edge_index, num_nodes, edge_weight, cache=cache)

    num_dense_layers = len(kernels)

    h = x
    for i, (kernel, bias) in enumerate(zip(kernels, biases)):
        if training and dense_drop_rate > 0.0:
            h = tf.compat.v2.nn.dropout(h, dense_drop_rate)
        h = h @ kernel + bias
        if dense_activation is not None and i < num_dense_layers - 1:
            h = dense_activation(h)

    if training and edge_drop_rate > 0.0:
        normed_edge_weight = tf.compat.v2.nn.dropout(normed_edge_weight, edge_drop_rate)

    prop_h = h

    for i in range(num_iterations):
        prop_h = aggregate_neighbors(
            prop_h, updated_edge_index, normed_edge_weight,
            gcn_mapper,
            sum_reducer,
            identity_updater,
            num_nodes=num_nodes
        )
        prop_h = prop_h * (1.0 - alpha) + h * alpha

    if activation is not None:
        prop_h = activation(prop_h)

    return prop_h

