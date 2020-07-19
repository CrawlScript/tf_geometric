import tensorflow as tf
from tf_geometric.utils.graph_utils import remove_self_loop_edge, add_self_loop_edge, get_laplacian
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_updater


def normalization(edge_index, num_nodes, edge_weight, lambda_max, normalization_type):
    edge_index, edge_weight = remove_self_loop_edge(edge_index, edge_weight)

    updated_edge_index, updated_edge_weight = get_laplacian(edge_index, edge_weight, normalization_type,
                                                            num_nodes)

    scaled_edge_weight = (2.0 * updated_edge_weight) / lambda_max


    assert edge_weight is not None

    return updated_edge_index, scaled_edge_weight


# def get_laplacian(edge_index, edge_weight, normalization_type, num_nodes, fill_weight=1.0):
#     if normalization_type is not None:
#         assert normalization_type in ['sym', 'rw']
#
#     edge_index, edge_weight = remove_self_loop_edge(edge_index, edge_weight)
#
#     if edge_weight is None:
#         edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)
#
#     row, col = edge_index
#     deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=num_nodes)
#
#     ##L = D - A
#     if normalization_type is None:
#         edge_index, _ = add_self_loop_edge(edge_index, num_nodes, fill_weight=fill_weight)
#         edge_weight = deg - edge_weight
#
#     ## L^ = D^{-1/2}LD^{-1/2}
#     elif normalization_type == 'sym':
#         deg_inv_sqrt = tf.pow(deg, -0.5)
#         deg_inv_sqrt = tf.where(
#             tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
#             tf.zeros_like(deg_inv_sqrt),
#             deg_inv_sqrt
#         )
#
#         normed_edge_weight = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)
#         edge_index, tmp = add_self_loop_edge(edge_index, num_nodes, edge_weight=normed_edge_weight, fill_weight=fill_weight)
#
#         assert tmp is not None
#         edge_weight = tmp
#     ##L^ = D^{-1}L
#     else:
#         deg_inv = 1.0 / deg
#         deg_inv = tf.where(
#             tf.math.logical_or(tf.math.is_inf(deg_inv), tf.math.is_nan(deg_inv)),
#             tf.zeros_like(deg_inv),
#             deg_inv
#         )
#
#         normed_edge_weight = tf.gather(deg_inv, row) * edge_weight
#
#         edge_index, tmp = add_self_loop_edge(edge_index, num_nodes, edge_weight=-normed_edge_weight, fill_weight=fill_weight)
#
#         assert tmp is not None
#         edge_weight = tmp
#
#     return edge_index, edge_weight


def gcn_mapper(repeated_x, neighbor_x, edge_weight=None):
    return neighbor_x * tf.expand_dims(edge_weight, 1)


def chebnet(x, edge_index, edge_weight):
    h = aggregate_neighbors(
        x, edge_index, edge_weight,
        gcn_mapper,
        sum_reducer,
        identity_updater
    )

    return h
