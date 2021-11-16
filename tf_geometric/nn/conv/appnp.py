# coding=utf-8

import tensorflow as tf
from tf_geometric.sparse.sparse_adj import SparseAdj
from tf_geometric.nn.conv.gcn import gcn_norm_adj
import tf_sparse as tfs


def appnp(x, edge_index, edge_weight, kernels, biases,
          dense_activation=tf.nn.relu, activation=None,
          num_iterations=10, alpha=0.1,
          dense_drop_rate=0.0, edge_drop_rate=0.0, cache=None, training=False):

    """
    Functional API for Approximate Personalized Propagation of Neural Predictions (APPNP).

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param kernels: List[Tensor], shape of each Tensor: [num_features, num_output_features], weights
    :param biases: List[Tensor], shape of each Tensor: [num_output_features], biases
    :param dense_activation: Activation function to use for the dense layers,
        except for the last dense layer, which will not be activated.
    :param activation: Activation function to use for the output.
    :param num_iterations: Number of propagation power iterations.
    :param alpha: Teleport Probability.
    :param dense_drop_rate: Dropout rate for the input of every dense layer.
    :param edge_drop_rate: Dropout rate for the edges/adj used for propagation.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        To use @tf_utils.function with gcn, you should cache the noremd edge information before the first call of the gcn.

        - (1) If you're using OOP APIs tfg.layers.GCN:

              gcn_layer.build_cache_for_graph(graph)

        - (2) If you're using functional API tfg.nn.gcn:

              from tf_geometric.nn.conv.gcn import gcn_build_cache_for_graph
              gcn_build_cache_for_graph(graph)

    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    num_nodes = tfs.shape(x)[0]
    # updated_edge_index, normed_edge_weight = gcn_norm_edge(edge_index, num_nodes, edge_weight, cache=cache)
    sparse_adj = SparseAdj(edge_index, edge_weight, [num_nodes, num_nodes])
    normed_sparse_adj = gcn_norm_adj(sparse_adj, cache=cache)\
        .dropout(edge_drop_rate, training=training)

    num_dense_layers = len(kernels)

    h = x
    for i, (kernel, bias) in enumerate(zip(kernels, biases)):
        # SparseTensor is usually used for one-hot node features (For example, feature-less nodes.)
        if isinstance(h, tf.sparse.SparseTensor):
            h = tf.sparse.sparse_dense_matmul(h, kernel)
        else:
            h = h @ kernel

        if bias is not None:
            h += bias

        if i < num_dense_layers - 1:
            if dense_activation is not None:
                h = dense_activation(h)
            if training and dense_drop_rate > 0.0:
                h = tf.compat.v2.nn.dropout(h, dense_drop_rate)

    # new implementation based on SparseAdj
    # sparse_adj = SparseAdj(updated_edge_index, normed_edge_weight, [num_nodes, num_nodes])\
    #     .dropout(edge_drop_rate, training=training)

    prop_h = h

    for i in range(num_iterations):
        prop_h = normed_sparse_adj @ prop_h
        prop_h = prop_h * (1.0 - alpha) + h * alpha


    # # old implementation
    # if training and edge_drop_rate > 0.0:
    #     normed_edge_weight = tf.compat.v2.nn.dropout(normed_edge_weight, edge_drop_rate)
    #
    # prop_h = h
    #
    # for i in range(num_iterations):
    #     prop_h = aggregate_neighbors(
    #         prop_h, updated_edge_index, normed_edge_weight,
    #         gcn_mapper,
    #         sum_reducer,
    #         identity_updater,
    #         num_nodes=num_nodes
    #     )
    #     prop_h = prop_h * (1.0 - alpha) + h * alpha

    if activation is not None:
        prop_h = activation(prop_h)

    return prop_h

