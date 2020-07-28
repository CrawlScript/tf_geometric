# coding=utf-8

from tf_geometric.nn.conv.gcn import gcn_norm_edge, gcn_mapper
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_updater


def sgc(x, edge_index, edge_weight, K, kernel, bias=None, renorm=True, improved=False, cache=None):
    """
    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param K: Number of hops.(default: :obj:`1`)
    :param kernel: Tensor, shape: [num_features, num_output_features], weight.
    :param bias: Tensor, shape: [num_output_features], bias.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
    :return: Updated node features (x), shape: [num_nodes, num_features]
    """
    updated_edge_index, normed_edge_weight = gcn_norm_edge(edge_index, x.shape[0], edge_weight,
                                                           renorm, improved, cache)

    h = x
    for _ in range(K):
        h = aggregate_neighbors(
            h,
            updated_edge_index,
            normed_edge_weight,
            gcn_mapper,
            sum_reducer,
            identity_updater
        )

    h = h @ kernel

    if bias is not None:
        h += bias
    return h

