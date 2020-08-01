# coding=utf-8
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_mapper, identity_updater, \
    sum_updater



def gin_updater(x, reduced_neighbor_msg, eps):
    return x * (1 + eps) + reduced_neighbor_msg


def gin(x, edge_index, edge_weight, mlp_model, eps=0.0, activation=None, cache=None):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param eps: float, optional, (default: :obj:`0.`).
    :param mlp_model: A neural network (multi-layer perceptrons).
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param cache: A dict for caching A' for GIN. Different graph should not share the same cache dict.
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    h = aggregate_neighbors(
        x, edge_index, edge_weight,
        identity_mapper,
        sum_reducer,
        identity_updater
    )

    h = gin_updater(x, h, eps)

    h = mlp_model(h)

    if activation is not None:
        h = activation(h)

    return h
