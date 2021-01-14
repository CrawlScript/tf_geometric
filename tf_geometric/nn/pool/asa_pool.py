# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.pool.cluster_pool import cluster_pool

from tf_geometric.nn.conv.le_conv import le_conv
from tf_geometric.nn.kernel.map_reduce import aggregate_neighbors, sum_reducer, identity_updater

from tf_geometric.nn.kernel.segment import segment_softmax

from tf_geometric.nn import max_pool
from tf_geometric.nn.conv.gcn import gcn, gcn_mapper

from tf_geometric.data.graph import BatchGraph
from tf_geometric.nn.pool.topk_pool import topk_pool
from tf_geometric.utils.graph_utils import add_self_loop_edge


def asa_pool(x, edge_index, edge_weight, node_graph_index,
             feature_gnn,
             attention_gcn_kernel, attention_gcn_bias,
             attention_query_kernel, attention_query_bias,
             attention_score_kernel, attention_score_bias,
             score_self_kernel, score_aggr_self_kernel, score_aggr_neighbor_kernel,
             K=None, ratio=None,

             attention_gcn_activation=None,
             attention_query_pool=max_pool,
             score_activation=tf.nn.sigmoid,


             score_activation=None,
             drop_rate=0.0, training=None, cache=None):
    """
    Functional API for ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representation

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param node_graph_index: Tensor/NDArray, shape: [num_nodes], graph index for each node
    :param feature_gnn: A GNN model to learn pooled node features, [x, edge_index, edge_weight] => updated_x,
        where updated_x corresponds to high-order node features.
    :param K: Keep top K targets for each source
    :param ratio: Keep num_targets * ratio targets for each source
    :param score_activation: Activation to use for node_score before multiplying node_features with node_score
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
    :return: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
    """


    num_nodes = tf.shape(x)[0]
    num_graphs = tf.reduce_max(node_graph_index) + 1

    if cache is None:
        h = feature_gnn([x, edge_index, edge_weight], training=training)
    else:
        h = feature_gnn([x, edge_index, edge_weight], training=training, cache=cache)

    edge_index_with_self_loop, updated_edge_weight = add_self_loop_edge(edge_index, num_nodes=num_nodes, edge_weight=edge_weight)


    row_with_self_loop, col_with_self_loop = edge_index_with_self_loop[0], edge_index_with_self_loop[1]

    attention_h = gcn(x, edge_index, edge_weight, attention_gcn_kernel, attention_gcn_bias, activation=attention_gcn_activation)
    attention_query = attention_query_pool(attention_h, row_with_self_loop, num_graphs=num_graphs)
    attention_query = attention_query @ attention_query_kernel + attention_query_bias


    repeated_attention_query = tf.gather(attention_query, row_with_self_loop)
    repeated_attention_h = tf.gather(attention_h, col_with_self_loop)

    attention_score_h = tf.concat([repeated_attention_query, repeated_attention_h], axis=-1)
    attention_score = attention_score_h @ attention_score_kernel + attention_score_bias

    normed_attention_score = segment_softmax(attention_score, row_with_self_loop, num_graphs)
    if training and drop_rate > 0:
        normed_attention_score = tf.compat.v2.nn.dropout(normed_attention_score, rate=drop_rate)

    h = aggregate_neighbors(
        h, edge_index_with_self_loop, normed_attention_score,
        gcn_mapper,
        sum_reducer,
        identity_updater,
        num_nodes=num_nodes
    )

    node_score = le_conv(h, edge_index, edge_weight,
                    score_self_kernel, score_aggr_self_kernel,
                    score_aggr_neighbor_kernel, activation=None)


    topk_node_index = topk_pool(node_graph_index, node_score, K=K, ratio=ratio)
    topk_node_score = tf.gather(node_score, topk_node_index)
    if score_activation is not None:
        topk_node_score = score_activation(topk_node_score)

    h = tf.gather(h, topk_node_index)
    h *= topk_node_score

    num_clusters = tf.shape(topk_node_index)[0]
    # node->cluster
    cluster_reverse_index = tf.cast(tf.fill([num_clusters], -1), tf.int32)
    cluster_reverse_index = tf.tensor_scatter_nd_update(
        cluster_reverse_index,
        tf.expand_dims(topk_node_index, axis=-1),
        tf.range(num_clusters)
    )

    row, col = edge_index[0], edge_index[1]
    assign_row = tf.gather(cluster_reverse_index, row)
    assign_mask = tf.greater_equal(assign_row, 0)

    assign_row = tf.boolean_mask(assign_row, assign_mask)
    assign_col = tf.boolean_mask(col, assign_mask)
    assign_edge_index = tf.stack([assign_row, assign_col], axis=0)
    assign_edge_weight = tf.gather(topk_node_score, assign_row)

    # Coarsen in a large BatchGraph.
    pooled_x, pooled_edge_index, pooled_edge_weight = cluster_pool(
        None, edge_index_with_self_loop, edge_weight, assign_edge_index, assign_edge_weight,
        num_clusters, num_nodes=num_nodes)

    pooled_node_graph_index = tf.gather(node_graph_index, topk_node_index)

    return pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index
