# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.pool.cluster_pool import cluster_pool
from tf_geometric.utils.graph_utils import convert_dense_assign_to_edge


def diff_pool_coarsen(x, edge_index, edge_weight, node_graph_index, dense_assign,
                      num_nodes=None, num_clusters=None, num_graphs=None):
    """
    Coarsening method for DiffPool.
    Coarsen the input BatchGraph (graphs) based on cluster assignment of nodes and output pooled BatchGraph (graphs).
    Graphs should be modeled as a BatchGraph like format and each graph has the same number of clusters.

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param node_graph_index: Tensor/NDArray, shape: [num_nodes], graph index for each node
    :param dense_assign: Tensor, [num_nodes, num_clusters], cluster assignment matrix of nodes.
    :param num_nodes: Number of nodes, Optional, used for boosting performance.
    :param num_clusters: Number of clusters, Optional, used for boosting performance.
    :param num_graphs: Number of graphs, Optional, used for boosting performance.
    :return: Pooled BatchGraph: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
    """

    # Manually passing in num_nodes, num_clusters, and num_graphs can boost the performance
    dense_shape = tf.shape(dense_assign)

    if num_nodes is None:
        num_nodes = dense_shape[0]

    if num_clusters is None:
        num_clusters = dense_shape[1]

    if num_graphs is None:
        num_graphs = tf.reduce_max(node_graph_index) + 1

    if edge_weight is None:
        num_edges = tf.shape(edge_index)[-1]
        edge_weight = tf.ones([num_edges], dtype=tf.float32)

    assign_edge_index, assign_edge_weight = convert_dense_assign_to_edge(dense_assign, node_graph_index,
                                                                         num_nodes=num_nodes, num_clusters=num_clusters)

    # Coarsen in a large BatchGraph.
    # Here num_clusters is the sum of number of clusters of graphs in the BatchGraph
    pooled_x, pooled_edge_index, pooled_edge_weight = cluster_pool(x, edge_index, edge_weight, assign_edge_index, assign_edge_weight,
                                                                   num_clusters=num_clusters * num_graphs, num_nodes=num_nodes)

    pooled_node_graph_index = tf.reshape(tf.tile(tf.expand_dims(tf.range(num_graphs), axis=-1), [1, num_clusters]), -1)

    return pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index


def diff_pool(x, edge_index, edge_weight, node_graph_index,
              feature_gnn, assign_gnn,
              num_clusters, bias=None, activation=None, cache=None, training=None):
    """
    Functional API for DiffPool: "Hierarchical graph representation learning with differentiable pooling"

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param node_graph_index: Tensor/NDArray, shape: [num_nodes], graph index for each node
    :param feature_gnn: A GNN model to learn pooled node features, [x, edge_index, edge_weight] => updated_x,
        where updated_x corresponds to high-order node features.
    :param assign_gnn: A GNN model to learn cluster assignment for the pooling, [x, edge_index, edge_weight] => updated_x,
        where updated_x corresponds to the cluster assignment matrix.
    :param num_clusters: Number of clusters for pooling.
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: [pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index]
    """

    if edge_weight is None:
        num_edges = tf.shape(edge_index)[-1]
        edge_weight = tf.ones([num_edges], dtype=tf.float32)

    num_nodes = tf.shape(x)[0]
    num_graphs = tf.reduce_max(node_graph_index) + 1

    if cache is None:
        assign_logits = assign_gnn([x, edge_index, edge_weight], training=training)
        h = feature_gnn([x, edge_index, edge_weight], training=training)
    else:
        assign_logits = assign_gnn([x, edge_index, edge_weight], training=training, cache=cache)
        h = feature_gnn([x, edge_index, edge_weight], training=training, cache=cache)

    assign_probs = tf.nn.softmax(assign_logits, axis=-1)

    pooled_h, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index = diff_pool_coarsen(
        h, edge_index, edge_weight, node_graph_index, assign_probs,
        num_nodes=num_nodes, num_clusters=num_clusters, num_graphs=num_graphs
    )

    if bias is not None:
        pooled_h += bias

    if activation is not None:
        pooled_h = activation(pooled_h)

    return pooled_h, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index