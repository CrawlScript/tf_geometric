# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.pool.cluster_pool import cluster_pool

from tf_geometric.nn.conv.le_conv import le_conv
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_updater, identity_mapper, \
    max_reducer

from tf_geometric.nn.kernel.segment import segment_softmax

from tf_geometric.nn import max_pool
from tf_geometric.nn.conv.gcn import gcn, gcn_mapper

from tf_geometric.data.graph import BatchGraph
from tf_geometric.nn.pool.topk_pool import topk_pool
from tf_geometric.utils.graph_utils import add_self_loop_edge, remove_self_loop_edge


def asap(x, edge_index, edge_weight, node_graph_index,
         attention_gcn_kernel, attention_gcn_bias,
         attention_query_kernel, attention_query_bias,
         attention_score_kernel, attention_score_bias,
         le_conv_self_kernel, le_conv_self_bias,
         le_conv_aggr_self_kernel, le_conv_aggr_self_bias,
         le_conv_aggr_neighbor_kernel, le_conv_aggr_neighbor_bias,
         k=None, ratio=None,
         le_conv_activation=tf.nn.sigmoid,
         drop_rate=0.0, training=None, cache=None):
    """
    Functional API for ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representation

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param node_graph_index: Tensor/NDArray, shape: [num_nodes], graph index for each node
    :param k: Keep top k targets for each source
    :param ratio: Keep num_targets * ratio targets for each source
    :param le_conv_activation: Activation to use for node_score before multiplying node_features with node_score
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
    :return: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
    """

    num_nodes = tf.shape(x)[0]
    # num_graphs = tf.reduce_max(node_graph_index) + 1

    edge_index, edge_weight = remove_self_loop_edge(edge_index, edge_weight)
    edge_index_with_self_loop, edge_weight_with_self_loop = add_self_loop_edge(edge_index, num_nodes=num_nodes,
                                                                               edge_weight=edge_weight)

    row_with_self_loop, col_with_self_loop = edge_index_with_self_loop[0], edge_index_with_self_loop[1]

    attention_h = gcn(x, edge_index, edge_weight, attention_gcn_kernel, attention_gcn_bias, cache=cache)

    # max_pool -> query
    attention_query = aggregate_neighbors(
        attention_h, edge_index_with_self_loop, None,
        mapper=identity_mapper,
        reducer=max_reducer,
        updater=identity_updater,
        num_nodes=num_nodes
    )

    attention_query = attention_query @ attention_query_kernel + attention_query_bias

    repeated_attention_query = tf.gather(attention_query, row_with_self_loop)
    repeated_attention_h = tf.gather(attention_h, col_with_self_loop)

    attention_score_h = tf.concat([repeated_attention_query, repeated_attention_h], axis=-1)
    attention_score = attention_score_h @ attention_score_kernel + attention_score_bias
    attention_score = tf.nn.leaky_relu(attention_score, alpha=0.2)

    normed_attention_score = segment_softmax(attention_score, row_with_self_loop, num_nodes)
    if training and drop_rate > 0:
        normed_attention_score = tf.compat.v2.nn.dropout(normed_attention_score, rate=drop_rate)

    # nodes are clusters
    cluster_h = aggregate_neighbors(
        x, edge_index_with_self_loop, tf.reshape(normed_attention_score, [-1]),
        gcn_mapper,
        sum_reducer,
        identity_updater,
        num_nodes=num_nodes
    )

    node_score = le_conv(cluster_h, edge_index, edge_weight,
                         le_conv_self_kernel, le_conv_self_bias,
                         le_conv_aggr_self_kernel, le_conv_aggr_self_bias,
                         le_conv_aggr_neighbor_kernel, le_conv_aggr_neighbor_bias,
                         activation=None)

    topk_node_index = topk_pool(node_graph_index, node_score, k=k, ratio=ratio)
    topk_node_score = tf.gather(node_score, topk_node_index)
    if le_conv_activation is not None:
        topk_node_score = le_conv_activation(topk_node_score)

    pooled_x = tf.gather(cluster_h, topk_node_index) * topk_node_score

    num_clusters = tf.shape(topk_node_index)[0]
    # node->cluster
    cluster_reverse_index = tf.cast(tf.fill([num_nodes], -1), tf.int32)
    cluster_reverse_index = tf.tensor_scatter_nd_update(
        cluster_reverse_index,
        tf.expand_dims(topk_node_index, axis=-1),
        tf.range(num_clusters)
    )

    # row, col = edge_index[0], edge_index[1]
    assign_row = tf.gather(cluster_reverse_index, row_with_self_loop)
    assign_mask = tf.greater_equal(assign_row, 0)

    assign_row = tf.boolean_mask(assign_row, assign_mask)
    assign_col = tf.boolean_mask(col_with_self_loop, assign_mask)
    assign_edge_index = tf.stack([assign_row, assign_col], axis=0)

    assign_edge_weight = tf.boolean_mask(normed_attention_score, assign_mask)
    assign_edge_weight = tf.reshape(assign_edge_weight, [-1])
    assign_edge_weight = tf.stop_gradient(assign_edge_weight)

    # Coarsen in a large BatchGraph.
    _, pooled_edge_index, pooled_edge_weight = cluster_pool(
        None, edge_index_with_self_loop, edge_weight_with_self_loop, assign_edge_index, assign_edge_weight,
        num_clusters, num_nodes=num_nodes)

    pooled_edge_index, pooled_edge_weight = remove_self_loop_edge(pooled_edge_index, pooled_edge_weight)
    pooled_edge_index, pooled_edge_weight = add_self_loop_edge(pooled_edge_index, num_clusters, pooled_edge_weight)

    pooled_node_graph_index = tf.gather(node_graph_index, topk_node_index)

    return pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index
