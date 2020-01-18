# coding=utf-8

import tensorflow as tf
from tf_geometric import Graph
from tf_geometric.nn.gnn.gcn import gcn_norm, gcn
from tf_geometric.layers.kernel.map_reduce import MapReduceGNN


class GCN(MapReduceGNN):

    def __init__(self, units, activation=None, use_cache=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.use_cache = use_cache

        self.acvitation = activation
        self.weight = None
        self.bias = None

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.weight = tf.Variable(tf.truncated_normal([num_features, self.units]))
        self.bias = tf.Variable(tf.zeros([self.units]))

    @classmethod
    def create_normed_edge_weight(cls, graph: Graph, use_cache=True):
        if use_cache and graph.cached_normed_edge_weight is not None:
            return graph.cached_normed_edge_weight
        else:
            normed_edge_weight = gcn_norm(graph.edge_index, graph.num_nodes, graph.edge_weight)
            if use_cache:
                graph.cached_normed_edge_weight = normed_edge_weight
            return normed_edge_weight

    def call(self, inputs, training=None, mask=None):
        x, edge_index, normed_edge_weight = inputs
        return gcn(x, edge_index, normed_edge_weight, self.weight, self.bias, activation=self.acvitation)