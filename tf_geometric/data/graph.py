# coding=utf-8
import tensorflow as tf
import numpy as np

from tf_geometric.utils.graph_utils import convert_edge_to_directed, compute_edge_mask_by_node_index
from tf_geometric.utils.union_utils import union_len, convert_union_to_numpy


class Graph(object):
    """
    A Graph object wrappers all the data of a graph,
    including node features, edge info (index and weight) and graph label
    """

    def __init__(self, x, edge_index, y=None, edge_weight=None):
        """

        :param x: Tensor/NDArray, shape: [num_nodes, num_features], node features
        :param edge_index: Tensor/NDArray, shape: [2, num_edges], edge information.
            Each column of edge_index (u, v) represents an directed edge from u to v.
            Note that it does not cover the edge from v to u. You should provide (v, u) to cover it.
            This is not convenient for users.
            Thus, we allow users to provide edge_index in undirected form and convert it later.
            That is, we can only provide (u, v) and convert it to (u, v) and (v, u) with `convert_edge_to_directed` method.
        :param y: Tensor/NDArray/None, any shape, graph label.
            If you want to use this object to construct a BatchGraph object, y cannot be a scalar Tensor.
        :param edge_weight: Tensor/NDArray/None, shape: [num_edges]
        """

        self.x = Graph.cast_x(x)
        self.edge_index = Graph.cast_edge_index(edge_index)
        self.y = y
        self.cache = {}

        if edge_weight is not None:
            self.edge_weight = self.cast_edge_weight(edge_weight)
        else:
            self.edge_weight = np.full([self.num_edges], 1.0, dtype=np.float32)
            if tf.is_tensor(self.x):
                self.edge_weight = tf.convert_to_tensor(self.edge_weight)

    @classmethod
    def cast_edge_index(cls, x):
        if isinstance(x, list):
            x = np.array(x).astype(np.int32)
        elif isinstance(x, np.ndarray):
            x = x.astype(np.int32)
        elif tf.is_tensor(x):
            x = tf.cast(x, tf.int32)
        return x

    @classmethod
    def cast_edge_weight(cls, edge_weight):
        if isinstance(edge_weight, list):
            edge_weight = np.array(edge_weight).astype(np.float32)
        elif isinstance(edge_weight, np.ndarray):
            edge_weight = edge_weight.astype(np.float32)
        elif tf.is_tensor(edge_weight):
            edge_weight = tf.cast(edge_weight, tf.float32)
        return edge_weight

    @classmethod
    def cast_x(cls, x):
        if isinstance(x, list):
            x = np.array(x)

        if isinstance(x, np.ndarray) and x.dtype == np.float64:
            x = x.astype(np.float32)
        elif tf.is_tensor(x) and x.dtype == tf.float64:
            x = tf.cast(x, tf.float32)

        return x

    @property
    def num_nodes(self):
        """
        Number of nodes
        :return: Number of nodes
        """
        return union_len(self.x)

    @property
    def num_edges(self):
        """
        Number of edges
        :return: Number of edges
        """

        if len(self.edge_index) == 0:
            return 0
        else:
            return len(self.edge_index[0])

        # if tf.is_tensor(self.edge_index):
        #     return self.edge_index.shape.as_list()[1]
        # else:
        #     return len(self.edge_index[0])

    @property
    def num_features(self):
        """
        Number of node features
        :return: Number of node features
        """
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

    def __repr__(self):
        return self.__str__()

    def _convert_data_to_tensor(self, keys):
        for key in keys:
            data = getattr(self, key)

            if data is not None and not tf.is_tensor(data):
                setattr(self, key, tf.convert_to_tensor(data))
        return self

    def convert_data_to_tensor(self):
        """
        Convert all graph data into Tensors. All corresponding properties will be replaces by their Tensor versions.
        :return: The Graph object itself.
        """
        return self._convert_data_to_tensor(["x", "edge_index", "edge_weight", "y"])

    def convert_edge_to_directed(self):
        """

        Each column of edge_index (u, v) represents an directed edge from u to v.
        Note that it does not cover the edge from v to u. You should provide (v, u) to cover it.
        This is not convenient for users.
        Thus, we allow users to provide edge_index in undirected form and convert it later.
        That is, we can only provide (u, v) and convert it to (u, v) and (v, u) with `convert_edge_to_directed` method.
        :return:
        """
        self.edge_index, [self.edge_weight] = convert_edge_to_directed(self.edge_index, [self.edge_weight])
        return self

    def sample_new_graph_by_node_index(self, sampled_node_index):
        """

        :param sampled_node_index: Tensor/NDArray, shape: [num_sampled_nodes]
        :return: A new cloned graph where nodes that are not in sampled_node_index are removed,
            as well as the associated information, such as edges.
        """
        is_batch_graph = isinstance(self, BatchGraph)

        x = self.x
        edge_index = self.edge_index
        y = self.y
        edge_weight = self.edge_weight
        if is_batch_graph:
            node_graph_index = self.node_graph_index
            edge_graph_index = self.edge_graph_index

        def sample_common_data(data):
            if data is not None:
                data_is_tensor = tf.is_tensor(data)
                if data_is_tensor:
                    data = tf.gather(data, sampled_node_index)
                else:
                    data = convert_union_to_numpy(data)
                    data = data[sampled_node_index]

                if data_is_tensor:
                    data = tf.convert_to_tensor(data)
            return data

        x = sample_common_data(x)
        y = sample_common_data(y)
        if is_batch_graph:
            node_graph_index = sample_common_data(node_graph_index)

        if edge_index is not None:

            sampled_node_index = convert_union_to_numpy(sampled_node_index)

            edge_index_is_tensor = tf.is_tensor(edge_index)
            edge_index = convert_union_to_numpy(edge_index)
            edge_mask = compute_edge_mask_by_node_index(edge_index, sampled_node_index)

            edge_index = edge_index[:, edge_mask]
            row, col = edge_index

            max_sampled_node_index = np.max(sampled_node_index) + 1
            new_node_range = list(range(len(sampled_node_index)))
            reverse_index = np.full([max_sampled_node_index + 1], -1, dtype=np.int32)
            reverse_index[sampled_node_index] = new_node_range

            row = reverse_index[row]
            col = reverse_index[col]
            edge_index = np.stack([row, col], axis=0)
            if edge_index_is_tensor:
                edge_index = tf.convert_to_tensor(edge_index)

            def sample_by_edge_mask(data):
                if data is not None:
                    data_is_tensor = tf.is_tensor(data)
                    data = convert_union_to_numpy(data)
                    data = data[edge_mask]
                    if data_is_tensor:
                        data = tf.convert_to_tensor(data)
                return data

            edge_weight = sample_by_edge_mask(edge_weight)
            if is_batch_graph:
                edge_graph_index = sample_by_edge_mask(edge_graph_index)

        if is_batch_graph:
            return BatchGraph(x=x, edge_index=edge_index, node_graph_index=node_graph_index,
                              edge_graph_index=edge_graph_index, y=y, edge_weight=edge_weight)
        else:
            return Graph(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)


class BatchGraph(Graph):
    """
    Batch graph wrap a batch of graphs into a single graph, where each nodes has an unique index and a graph index.
    The node_graph_index is the index of the corresponding graph for each node in the batch.
    The edge_graph_index is the index of the corresponding edge for each node in the batch.
    """

    def __init__(self, x, edge_index, node_graph_index, edge_graph_index,
                 y=None, edge_weight=None, graphs=None):
        """

        :param x: Tensor/NDArray, shape: [num_nodes, num_features], node features
        :param edge_index: Tensor/NDArray, shape: [2, num_edges], edge information.
            Each column of edge_index (u, v) represents an directed edge from u to v.
            Note that it does not cover the edge from v to u. You should provide (v, u) to cover it.
        :param node_graph_index: Tensor/NDArray, shape: [num_nodes], graph index for each node
        :param edge_graph_index: Tensor/NDArray/None, shape: [num_edges], graph index for each edge
        :param y: Tensor/NDArray/None, any shape, graph label.
            If you want to use this object to construct a BatchGraph object, y cannot be a scalar Tensor.
        :param edge_weight: Tensor/NDArray/None, shape: [num_edges]
        :param graphs: list[Graph], original graphs
        """
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

    def convert_data_to_tensor(self):
        """
        Convert all graph data into Tensors. All corresponding properties will be replaces by their Tensor versions.
        :return: The Graph object itself.
        """
        return self._convert_data_to_tensor(["x", "edge_index", "edge_weight", "y",
                                             "node_graph_index", "edge_graph_index"])

    def convert_edge_to_directed(self):
        """

        Each column of edge_index (u, v) represents an directed edge from u to v.
        Note that it does not cover the edge from v to u. You should provide (v, u) to cover it.
        This is not convenient for users.
        Thus, we allow users to provide edge_index in undirected form and convert it later.
        That is, we can only provide (u, v) and convert it to (u, v) and (v, u) with `convert_edge_to_directed` method.
        :return:
        """
        self.edge_index, [self.edge_weight, self.edge_graph_index] = \
            convert_edge_to_directed(self.edge_index, [self.edge_weight, self.edge_graph_index])
        return self



