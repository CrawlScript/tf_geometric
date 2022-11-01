# coding=utf-8
import types
import warnings

import tensorflow as tf
import numpy as np
import tf_sparse as tfs

from tf_geometric.utils.graph_utils import convert_edge_to_directed, compute_edge_mask_by_node_index
from tf_geometric.utils.tf_sparse_utils import sparse_gather_sub
from tf_geometric.utils.union_utils import union_len, convert_union_to_numpy


def _get_shape(data):
    return None if data is None else data.shape


class Graph(object):

    tensor_spec_edge_index = tf.TensorSpec(shape=[2, None], dtype=tf.int32)
    tensor_spec_edge_weight = tf.TensorSpec(shape=[None], dtype=tf.float32)

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

        self._x = Graph.cast_x(x)
        self.edge_index = Graph.cast_edge_index(edge_index)
        self.y = Graph.cast_y(y)
        self.cache = {}

        if edge_weight is not None:
            self.edge_weight = self.cast_edge_weight(edge_weight)
        else:
            if tf.is_tensor(self.edge_index):
                self.edge_weight = tf.ones([self.num_edges], dtype=tf.float32)
            else:
                self.edge_weight = np.ones([self.num_edges], dtype=np.float32)

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

    @classmethod
    def cast_y(cls, y):
        if isinstance(y, list):
            y = np.array(y)
        return y

    
    @property
    def tensor_spec_x(self):
        return tf.TensorSpec(
            shape=[None, self.x.shape[-1]],
            dtype=tf.float32
        )

    @property
    def tensor_spec_y(self):
        if tf.is_tensor(self.y):
            dtype = self.y.dtype
        elif self.y.dtype == np.float32:
            dtype = tf.float32
        elif self.y.dtype == np.float64:
            dtype = tf.float64
        elif self.y.dtype == np.int32:
            dtype = tf.int32
        elif self.y.dtype == np.int64:
            dtype = tf.int64
        else:
            dtype = self.y.dtype

        return tf.TensorSpec(
            shape=[None] + list(self.y.shape)[1:],
            dtype=dtype
        )


    # @classmethod
    # def cast_y(cls, y):
    #     if y is None:
    #         return y
    #
    #     if isinstance(y, list):
    #         y = np.array(y)
    #
    #     if isinstance(y, np.ndarray) and y.dtype == np.float64:
    #         y = y.astype(np.float32)
    #     elif tf.is_tensor(y) and y.dtype == tf.float64:
    #         y = tf.cast(y, tf.float32)
    #
    #     return y

    @property
    def x(self):
        if isinstance(self._x, types.FunctionType):
            return self._x()
        else:
            return self._x

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
        if tf.is_tensor(self.edge_index):
            shape = tf.shape(self.edge_index)

            def return_empty_num_edges():
                return 0

            def return_common_num_edges():
                return shape[1]

            return tf.cond(shape[0] == 0, return_empty_num_edges, return_common_num_edges)

        else:
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

    # def get_shape(self, data):
    #     return None if data is None else data.shape

    def get_shape_desc(self):
        return "Graph Shape: x => {}\tedge_index => {}\ty => {}".format(
            _get_shape(self.x),
            _get_shape(self.edge_index),
            _get_shape(self.y)
        )

    def __str__(self):
        return self.get_shape_desc()

    def __repr__(self):
        return self.__str__()

    def adj(self):
        num_nodes = self.num_nodes
        return tfs.SparseMatrix(self.edge_index, self.edge_weight, shape=[num_nodes, num_nodes])

    def _inplace_convert_data_to_tensor(self, keys):
        for key in keys:
            data = getattr(self, key)

            if data is not None and not tf.is_tensor(data) and not isinstance(data, types.FunctionType):
                setattr(self, key, tf.convert_to_tensor(data))

        return self

    def convert_data_to_tensor(self, inplace=False):
        """
        Convert all graph data into Tensors. All corresponding properties will be replaces by their Tensor versions.

        :return: The Graph object itself.
        """
        if inplace:
            graph = self
        else:
            graph = Graph(self._x, self.edge_index, y=self.y, edge_weight=self.edge_weight)

        graph._inplace_convert_data_to_tensor(["_x", "edge_index", "edge_weight", "y"])
        return graph

    def to_directed(self, merge_mode="sum", inplace=False):
        """

        Each column of edge_index (u, v) represents an directed edge from u to v.
        Note that it does not cover the edge from v to u. You should provide (v, u) to cover it.
        This is not convenient for users.
        Thus, we allow users to provide edge_index in undirected form and convert it later.
        That is, we can simply provide (u, v) and convert it to (u, v) and (v, u) with `convert_edge_to_directed` method.

        :return: If inplace is False, return a new graph with directed edges. Else, return the current graph with directed edges.
        """

        edge_index, [edge_weight] = convert_edge_to_directed(self.edge_index, [self.edge_weight],
                                                             merge_modes=[merge_mode])
        if inplace:
            self.edge_index, self.edge_weight = edge_index, edge_weight
            return self
        else:
            return Graph(self.x, edge_index, y=self.y, edge_weight=edge_weight)

    def convert_edge_to_directed(self, merge_mode="sum"):
        """

        Each column of edge_index (u, v) represents an directed edge from u to v.
        Note that it does not cover the edge from v to u. You should provide (v, u) to cover it.
        This is not convenient for users.
        Thus, we allow users to provide edge_index in undirected form and convert it later.
        That is, we can only provide (u, v) and convert it to (u, v) and (v, u) with `convert_edge_to_directed` method.

        .. deprecated:: 0.0.84
            Use ``to_directed(inplace=True)`` instead.

        :return:
        """

        warnings.warn(
            "'Graph.convert_edge_to_directed(self, merge_mode)' is deprecated, use 'Graph.to_directed(inplace=True)' instead",
            DeprecationWarning)

        return self.to_directed(merge_mode, inplace=True)

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
                    if isinstance(data, (tf.sparse.SparseTensor, tfs.SparseMatrix)):
                        data = sparse_gather_sub(data, sampled_node_index)
                    else:
                        data = tf.gather(data, sampled_node_index)
                else:
                    data = convert_union_to_numpy(data)
                    data = data[sampled_node_index]

                # if data_is_tensor:
                #     data = tf.convert_to_tensor(data)
            return data

        x = sample_common_data(x)
        y = sample_common_data(y)
        if is_batch_graph:
            node_graph_index = sample_common_data(node_graph_index)

        if edge_index is not None:

            # sampled_node_index = convert_union_to_numpy(sampled_node_index)
            # edge_index = convert_union_to_numpy(edge_index)

            edge_index_is_tensor = tf.is_tensor(edge_index)
            if not edge_index_is_tensor:
                edge_index = tf.convert_to_tensor(edge_index, dtype=tf.int32)

            edge_mask = compute_edge_mask_by_node_index(edge_index, sampled_node_index)
            edge_index = tf.boolean_mask(edge_index, edge_mask, axis=1)

            row, col = edge_index[0], edge_index[1]

            max_sampled_node_index = tf.reduce_max(sampled_node_index) + 1
            num_sampled_nodes = tf.shape(sampled_node_index)[0]
            new_node_range = tf.range(num_sampled_nodes)

            reverse_index = tf.cast(tf.fill([max_sampled_node_index + 1], -1), tf.int32)
            reverse_index = tf.tensor_scatter_nd_update(reverse_index, tf.expand_dims(sampled_node_index, axis=-1),
                                                        new_node_range)

            row = tf.gather(reverse_index, row)
            col = tf.gather(reverse_index, col)

            edge_index = tf.stack([row, col], axis=0)
            if not edge_index_is_tensor:
                edge_index = edge_index.numpy()

            def sample_by_edge_mask(data):
                if data is not None:
                    data_is_tensor = tf.is_tensor(data)
                    data = tf.boolean_mask(data, edge_mask)
                    if not data_is_tensor:
                        data = data.numpy()
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
        if tf.is_tensor(self.node_graph_index):
            return tf.reduce_max(self.node_graph_index) + 1
        else:
            return np.max(self.node_graph_index) + 1

    def reorder(self):
        node_sort_index = tf.argsort(self.node_graph_index)
        node_graph_index = tf.gather(self.node_graph_index, node_sort_index)
        x = tf.gather(self.x, node_sort_index)
        if self.y is None:
            y = None
        else:
            y = tf.gather(self.y, node_sort_index)

        edge_sort_index = tf.argsort(self.edge_graph_index)
        edge_graph_index = tf.gather(self.edge_graph_index, edge_sort_index)
        edge_index = tf.gather(self.edge_index, edge_sort_index, axis=1)

        if self.edge_weight is None:
            edge_weight = None
        else:
            edge_weight = tf.gather(self.edge_weight, edge_sort_index)

        return BatchGraph(x, edge_index, node_graph_index, edge_graph_index, y=y, edge_weight=edge_weight)


    def to_graphs(self):
        batch_graph = self.reorder()
        # num_nodes_list = tf.math.segment_sum(tf.ones([self.num_nodes]), self.node_graph_index)
        num_graphs = batch_graph.num_graphs
        num_nodes_list = tf.math.unsorted_segment_sum(tf.ones([batch_graph.num_nodes]), batch_graph.node_graph_index, num_graphs)

        num_nodes_before_graph = tf.concat([
            tf.zeros([1]),
            tf.math.cumsum(num_nodes_list)
        ], axis=0).numpy().astype(np.int32).tolist()

        # num_edges_list = tf.math.segment_sum(tf.ones([self.num_edges]), self.edge_graph_index)
        num_edges_list = tf.math.unsorted_segment_sum(tf.ones([batch_graph.num_edges]), batch_graph.edge_graph_index, num_graphs)
        num_edges_before_graph = tf.concat([
            tf.zeros([1]),
            tf.math.cumsum(num_edges_list)
        ], axis=0).numpy().astype(np.int32).tolist()

        graphs = []
        for i in range(batch_graph.num_graphs):
            if isinstance(batch_graph.x, tf.sparse.SparseTensor):
                x = tf.sparse.slice(
                    batch_graph.x,
                    [num_nodes_before_graph[i], 0],
                    [num_nodes_before_graph[i + 1] - num_nodes_before_graph[i], tf.shape(batch_graph.x)[-1]]
                )
            else:
                x = batch_graph.x[num_nodes_before_graph[i]: num_nodes_before_graph[i + 1]]

            if batch_graph.y is None:
                y = None
            else:
                y = batch_graph.y[num_nodes_before_graph[i]: num_nodes_before_graph[i + 1]]

            edge_index = batch_graph.edge_index[:, num_edges_before_graph[i]:num_edges_before_graph[i + 1]] - \
                         num_nodes_before_graph[i]

            if batch_graph.edge_weight is None:
                edge_weight = None
            else:
                edge_weight = batch_graph.edge_weight[num_edges_before_graph[i]:num_edges_before_graph[i + 1]]

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

        if tf.is_tensor(graphs[0].edge_index):
            for i, graph in enumerate(graphs):
                node_graph_index_list.append(tf.fill([graph.num_nodes], i))
            node_graph_index = tf.concat(node_graph_index_list, axis=0)
            node_graph_index = tf.cast(node_graph_index, tf.int32)
        else:
            for i, graph in enumerate(graphs):
                node_graph_index_list.append(np.full([graph.num_nodes], i, dtype=np.int32))
            node_graph_index = np.concatenate(node_graph_index_list, axis=0)

        return node_graph_index

    @classmethod
    def build_edge_graph_index(cls, graphs):
        edge_graph_index_list = []
        if tf.is_tensor(graphs[0].edge_index):
            for i, graph in enumerate(graphs):
                edge_graph_index_list.append(tf.fill([graph.num_edges], i))
            edge_graph_index = tf.concat(edge_graph_index_list, axis=0)
            edge_graph_index = tf.cast(edge_graph_index, tf.int32)
        else:
            for i, graph in enumerate(graphs):
                edge_graph_index_list.append(np.full([graph.num_edges], i, dtype=np.int32))
            edge_graph_index = np.concatenate(edge_graph_index_list, axis=0)

        return edge_graph_index

    @classmethod
    def build_x(cls, graphs):
        x_list = [graph.x for graph in graphs]
        first_x = x_list[0]
        if tf.is_tensor(first_x):
            if isinstance(first_x, tfs.SparseMatrix):
                return tfs.concat(x_list, axis=0)
            elif isinstance(first_x, tf.sparse.SparseTensor):
                return tf.sparse.concat(0, x_list)
            else:
                return tf.concat(x_list, axis=0)
        else:
            return np.concatenate(x_list, axis=0)

    @classmethod
    def build_edge_index(cls, graphs):
        edge_index_list = []
        num_history_nodes = 0
        for i, graph in enumerate(graphs):
            edge_index_list.append(graph.edge_index + num_history_nodes)
            num_history_nodes += graph.num_nodes

        if tf.is_tensor(graphs[0].edge_index):
            return tf.concat(edge_index_list, axis=1)
        else:
            return np.concatenate(edge_index_list, axis=1)

    @classmethod
    def build_edge_weight(cls, graphs):
        if graphs[0].edge_weight is None:
            return None
        elif tf.is_tensor(graphs[0].edge_weight):
            return tf.concat([
                graph.edge_weight for graph in graphs
            ], axis=0)
        else:
            return np.concatenate([
                graph.edge_weight for graph in graphs
            ], axis=0)

    @classmethod
    def build_y(cls, graphs):
        if graphs[0].y is None:
            return None
        elif tf.is_tensor(graphs[0].y):
            return tf.concat([
                graph.y for graph in graphs
            ], axis=0)
        else:
            return np.concatenate([
                graph.y for graph in graphs
            ], axis=0)

    def convert_data_to_tensor(self, inplace=False):
        """
        Convert all graph data into Tensors. All corresponding properties will be replaces by their Tensor versions.

        :return: The Graph object itself.
        """
        if inplace:
            graph = self
        else:
            graph = BatchGraph(
                self._x, self.edge_index, 
                self.node_graph_index, self.edge_graph_index,
                y=self.y, edge_weight=self.edge_weight, graphs=self.graphs
                )
        return graph._inplace_convert_data_to_tensor(["_x", "edge_index", "edge_weight", "y",
                                                    "node_graph_index", "edge_graph_index"])

    def to_directed(self, merge_mode="sum", inplace=False):
        """

        Each column of edge_index (u, v) represents an directed edge from u to v.
        Note that it does not cover the edge from v to u. You should provide (v, u) to cover it.
        This is not convenient for users.
        Thus, we allow users to provide edge_index in undirected form and convert it later.
        That is, we can simply provide (u, v) and convert it to (u, v) and (v, u) with `convert_edge_to_directed` method.

        :return: If inplace is False, return a new graph with directed edges. Else, return the current graph with directed edges.
        """

        edge_index, [edge_weight, edge_graph_index] = \
            convert_edge_to_directed(self.edge_index, [self.edge_weight, self.edge_graph_index],
                                     merge_modes=[merge_mode, "max"])

        if inplace:
            self.edge_index, self.edge_weight, self.edge_graph_index = edge_index, edge_weight, edge_graph_index
            return self
        else:
            return BatchGraph(self.x, edge_index, self.node_graph_index, edge_graph_index, y=self.y, edge_weight=edge_weight)

    def convert_edge_to_directed(self, merge_mode="sum"):
        """

        Each column of edge_index (u, v) represents an directed edge from u to v.
        Note that it does not cover the edge from v to u. You should provide (v, u) to cover it.
        This is not convenient for users.
        Thus, we allow users to provide edge_index in undirected form and convert it later.
        That is, we can only provide (u, v) and convert it to (u, v) and (v, u) with `convert_edge_to_directed` method.

        .. deprecated:: 0.0.84
            Use ``to_directed(inplace=True)`` instead.

        :return:
        """

        warnings.warn(
            "'BatchGraph.convert_edge_to_directed(self, merge_mode)' is deprecated, use 'BatchGraph.to_directed(inplace=True)' instead",
            DeprecationWarning)

        return self.to_directed(merge_mode, inplace=True)


class HeteroDictGraph(object):

    def __init__(self, x_dict=None, edge_index_dict=None, y_dict=None, edge_weight_dict=None):
        if x_dict is None:
            x_dict = {}

        if edge_index_dict is None:
            edge_index_dict = {}

        if y_dict is None:
            y_dict = {}

        if edge_weight_dict is None:
            edge_weight_dict = {}

        self.x_dict = {node_type: Graph.cast_x(x) for node_type, x in x_dict.items()}
        self.edge_index_dict = {edge_type: Graph.cast_edge_index(edge_index)
                                for edge_type, edge_index in edge_index_dict.items()}
        self.y_dict = {y_type: Graph.cast_y(y) for y_type, y in y_dict.items()}
        self.edge_weight_dict = {edge_type: Graph.cast_edge_weight(edge_weight)
                                 for edge_type, edge_weight in edge_weight_dict}

        self.cache = {}

    def num_nodes(self, node_type=None):
        if node_type is not None:
            return tfs.shape(self.x_dict[node_type])[0]
        else:
            return {node_type: self.num_nodes(node_type=node_type) for node_type in self.x_dict}

    def num_edges(self, edge_type=None):
        if edge_type is not None:
            return tf.shape(self.edge_index_dict[edge_type])[1]
        else:
            return {edge_type: self.num_edges(edge_type) for edge_type in self.edge_index_dict}

    def get_shape_desc(self):

        x_shape_desc_dict = {
            node_type: _get_shape(x) for node_type, x in self.x_dict.items()
        }

        edge_index_shape_desc_dict = {
            edge_type: _get_shape(edge_index) for edge_type, edge_index in self.edge_index_dict.items()
        }

        y_shape_desc_dict = {
            y_type: _get_shape(y) for y_type, y in self.y_dict.items()
        }

        return "HeteroGraph Shape: \n\tx => {}\n\tedge_index => {}\n\ty => {}".format(
            x_shape_desc_dict, edge_index_shape_desc_dict, y_shape_desc_dict
        )

    def add_reversed_edges(self, reverse_prefix="r.", inplace=False):
        """
        """
        new_edge_index_dict = {**self.edge_index_dict}
        new_edge_weight_dict = {**self.edge_weight_dict}

        for edge_type, edge_index in self.edge_index_dict.items():
            edge_weight = self.edge_weight_dict[edge_type] if edge_type in self.edge_weight_dict else None
            reversed_edge_type = (edge_type[2], "{}{}".format(reverse_prefix, edge_type[1]), edge_type[0])

            if tf.is_tensor(edge_index):
                reversed_edge_index = tf.stack([edge_index[1], edge_index[0]], axis=0)
            else:
                reversed_edge_index = np.stack([edge_index[1], edge_index[0]], axis=0)

            new_edge_index_dict[reversed_edge_type] = reversed_edge_index
            if edge_weight is not None:
                new_edge_weight_dict[reversed_edge_type] = edge_weight

        if inplace:
            self.edge_index_dict, self.edge_weight_dict = new_edge_index_dict, new_edge_weight_dict
            return self
        else:
            return HeteroDictGraph(self.x_dict, new_edge_index_dict, y_dict=self.y_dict,
                                   edge_weight_dict=new_edge_weight_dict)

    def __str__(self):
        return self.get_shape_desc()

    def __repr__(self):
        return self.__str__()
