# coding=utf-8
import tensorflow as tf
import numpy as np

from tf_geometric.utils.graph_utils import convert_edge_to_directed


class Graph(object):
    def __init__(self, x, edge_index, y=None,
                 edge_weight=None):

        self.x = Graph.cast_x(x)
        self.edge_index = edge_index
        self.y = y
        self.cache = {}

        if edge_weight is not None:
            self.edge_weight = edge_weight
        else:
            self.edge_weight = np.full([len(self.edge_index[0])], 1.0, dtype=np.float32)
            if tf.is_tensor(self.x):
                self.edge_weight = tf.convert_to_tensor(self.edge_weight)

    @classmethod
    def cast_x(cls, x):
        if isinstance(x, list):
            x = np.array(x).astype(np.float32)
        elif isinstance(x, np.ndarray):
            x = x.astype(np.float32)
        elif tf.is_tensor(x):
            x = tf.cast(x, tf.float32)
        return x

    @property
    def num_nodes(self):
        return len(self.x)

    @property
    def num_edges(self):
        return len(self.edge_index[0])

    @property
    def num_features(self):
        return self.x.shape[-1]

    def get_shape(self, data):
        return None if data is None else data.shape

    def get_shape_desc(self):
        return "Graph Shape: x => {}\tedge_index => {}\ty => {}".format(
            self.get_shape(self.x),
            self.get_shape(self.edge_index),
            self.get_shape(self.y)
        )

    def __str__(self):
        return self.get_shape_desc()

    def convert_data_to_tensor(self):
        for key in ["x", "edge_index", "edge_weight", "y"]:
            data = getattr(self, key)

            if data is not None and not tf.is_tensor(data):
                setattr(self, key, tf.convert_to_tensor(data))
        return self

    def convert_edge_to_directed(self):
        self.edge_index, self.edge_weight = convert_edge_to_directed(self.edge_index, self.edge_weight)
        return self


class BatchGraph(Graph):
    """
    Batch graph wrap a batch of graphs into a single graph, where each nodes has an unique index and a graph index.
    The node_graph_index is the index of the corresponding graph for each node in the batch.
    The edge_graph_index is the index of the corresponding edge for each node in the batch.
    """

    def __init__(self, x, edge_index, node_graph_index, edge_graph_index, graphs=None, y=None, edge_weight=None):
        super().__init__(x, edge_index, y, edge_weight)
        self.node_graph_index = node_graph_index
        self.edge_graph_index = edge_graph_index
        self.graphs = graphs

    @property
    def num_graphs(self):
        return tf.reduce_max(self.node_graph_index) + 1

    def to_graphs(self):
        num_nodes_list = tf.math.segment_sum(tf.ones([self.num_nodes]), self.node_graph_index)
        num_nodes_before_graph = tf.concat([
            tf.zeros([1]),
            tf.math.cumsum(num_nodes_list)
        ], axis=0).numpy().astype(np.int32).tolist()

        num_edges_list = tf.math.segment_sum(tf.ones([self.num_edges]), self.edge_graph_index)
        num_edges_before_graph = tf.concat([
            tf.zeros([1]),
            tf.math.cumsum(num_edges_list)
        ], axis=0).numpy().astype(np.int32).tolist()

        graphs = []
        for i in range(self.num_graphs):
            x = self.x[num_nodes_before_graph[i]: num_edges_before_graph[i+1]]

            if self.y is None:
                y = None
            else:
                y = self.y[num_nodes_before_graph[i]: num_edges_before_graph[i+1]]

            edge_index = self.edge_index[:, num_edges_before_graph[i]:num_edges_before_graph[i+1]] - num_nodes_before_graph[i]

            if self.edge_weight is None:
                edge_weight = None
            else:
                edge_weight = self.edge_weight[num_edges_before_graph[i]:num_edges_before_graph[i+1]]

            graph = Graph(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)
            graphs.append(graph)
        return graphs



    @classmethod
    def from_graphs(cls, graphs):

        node_graph_index = BatchGraph.build_node_graph_index(graphs)
        edge_graph_index = BatchGraph.build_edge_graph_index(graphs)

        x = BatchGraph.build_x(graphs)
        edge_index = BatchGraph.build_edge_index(graphs)
        y = BatchGraph.build_y(graphs)
        edge_weight = BatchGraph.build_edge_weight(graphs)

        return BatchGraph(x=x, edge_index=edge_index,
                          node_graph_index=node_graph_index, edge_graph_index=edge_graph_index,
                          graphs=graphs, y=y, edge_weight=edge_weight)



    @classmethod
    def build_node_graph_index(cls, graphs):
        node_graph_index_list = []

        for i, graph in enumerate(graphs):
            node_graph_index_list.append(tf.fill([graph.num_nodes], i))

        node_graph_index = tf.concat(node_graph_index_list, axis=0)
        node_graph_index = tf.cast(node_graph_index, tf.int32)

        return node_graph_index


    @classmethod
    def build_edge_graph_index(cls, graphs):
        edge_graph_index_list = []

        for i, graph in enumerate(graphs):
            edge_graph_index_list.append(tf.fill([graph.num_edges], i))

        edge_graph_index = tf.concat(edge_graph_index_list, axis=0)
        edge_graph_index = tf.cast(edge_graph_index, tf.int32)

        return edge_graph_index


    @classmethod
    def build_x(cls, graphs):
        return tf.concat([
            graph.x for graph in graphs
        ], axis=0)

    @classmethod
    def build_edge_index(cls, graphs):
        edge_index_list = []
        num_history_nodes = 0
        for i, graph in enumerate(graphs):
            edge_index_list.append(graph.edge_index + num_history_nodes)
            num_history_nodes += graph.num_nodes

        return tf.concat(edge_index_list, axis=1)

    @classmethod
    def build_edge_weight(cls, graphs):
        if graphs[0].edge_weight is None:
            return None
        else:
            return tf.concat([
                graph.edge_weight for graph in graphs
            ], axis=0)


    @classmethod
    def build_y(cls, graphs):
        if graphs[0].y is None:
            return None
        else:
            return tf.concat([
                graph.y for graph in graphs
            ], axis=0)



