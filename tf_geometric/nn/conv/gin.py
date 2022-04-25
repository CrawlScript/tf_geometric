# coding=utf-8
import tensorflow as tf
# from tf_geometric.sparse.sparse_adj import SparseAdj
from tf_sparse import SparseMatrix


def gin_updater(x, reduced_neighbor_msg, eps):
    return x * (1.0 + eps) + reduced_neighbor_msg


def gin(x, edge_index, mlp_model, eps=0.0, training=None):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param mlp_model: A neural network (multi-layer perceptrons).
    :param eps: float, optional, (default: :obj:`0.`).
    :param training: Whether currently executing in training or inference mode.
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    # h = aggregate_neighbors(
    #     x, edge_index, None,
    #     identity_mapper,
    #     sum_reducer,
    #     identity_updater
    # )

    # h = gin_updater(x, h, eps)

    num_nodes = tf.shape(x)[0]
    sparse_adj = SparseMatrix(edge_index, shape=[num_nodes, num_nodes])

    neighbor_h = sparse_adj @ x
    h = x * (1.0 + eps) + neighbor_h
    h = mlp_model(h, training=training)

    return h
