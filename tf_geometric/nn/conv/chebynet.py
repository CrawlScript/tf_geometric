import tensorflow as tf
from tf_geometric.nn.conv.gcn import gcn_mapper
from tf_geometric.utils.graph_utils import remove_self_loop_edge, add_self_loop_edge, get_laplacian
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_updater


def chebynet_norm_edge(edge_index, num_nodes, edge_weight, lambda_max, normalization_type):
    edge_index, edge_weight = remove_self_loop_edge(edge_index, edge_weight)

    updated_edge_index, updated_edge_weight = get_laplacian(edge_index, edge_weight, normalization_type,
                                                            num_nodes)

    scaled_edge_weight = (2.0 * updated_edge_weight) / lambda_max

    assert edge_weight is not None

    return updated_edge_index, scaled_edge_weight


def chebynet(x, edge_index, edge_weight, K, lambda_max, kernel, bias=None, activation=None, normalization_type=None):
    num_nodes = x.shape[0]
    norm_edge_index, norm_edge_weight = chebynet_norm_edge(edge_index, num_nodes, edge_weight, lambda_max,
                                                           normalization_type=normalization_type)

    T0_x = x
    T1_x = x
    out = tf.matmul(T0_x, kernel[0])

    if K > 1:
        T1_x = aggregate_neighbors(x, norm_edge_index, norm_edge_weight, gcn_mapper, sum_reducer, identity_updater)
        out += tf.matmul(T1_x, kernel[1])

    for i in range(2, K):
        T2_x = aggregate_neighbors(T1_x, norm_edge_index, norm_edge_weight, gcn_mapper, sum_reducer,
                                   identity_updater)  ##L^T_{k-1}(L^)
        T2_x = 2.0 * T2_x - T0_x
        out += tf.matmul(T2_x, kernel[i])

        T0_x, T1_x = T1_x, T2_x

    if bias is not None:
        out += bias

    if activation is not None:
        out += activation(out)

    return out
