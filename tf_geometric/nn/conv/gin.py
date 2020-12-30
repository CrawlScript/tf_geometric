# coding=utf-8
<<<<<<< HEAD
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_mapper, identity_updater, \
    sum_updater

=======
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_mapper, identity_updater
>>>>>>> upstream/master


def gin_updater(x, reduced_neighbor_msg, eps):
    return x * (1.0 + eps) + reduced_neighbor_msg


<<<<<<< HEAD
def gin(x, edge_index, edge_weight, mlp_model, eps=0.0, activation=None, cache=None):
=======
def gin(x, edge_index, edge_weight, mlp_model, eps=0.0, training=None):
>>>>>>> upstream/master
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
<<<<<<< HEAD
    :param eps: float, optional, (default: :obj:`0.`).
    :param mlp_model: A neural network (multi-layer perceptrons).
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param cache: A dict for caching A' for GIN. Different graph should not share the same cache dict.
=======
    :param mlp_model: A neural network (multi-layer perceptrons).
    :param eps: float, optional, (default: :obj:`0.`).
    :param training: Whether currently executing in training or inference mode.
>>>>>>> upstream/master
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    h = aggregate_neighbors(
        x, edge_index, edge_weight,
        identity_mapper,
        sum_reducer,
        identity_updater
    )

    h = gin_updater(x, h, eps)

<<<<<<< HEAD
    h = mlp_model(h)

    if activation is not None:
        h = activation(h)
=======
    h = mlp_model(h, training=training)
>>>>>>> upstream/master

    return h
