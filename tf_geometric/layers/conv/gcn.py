# coding=utf-8

import tensorflow as tf
from tf_geometric import Graph
from tf_geometric.nn.conv.gcn import gcn_norm_edge, gcn
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
    def norm_edge(cls, graph: Graph, use_cache=True, improved=False):
        cache_key = "gcn"
        if use_cache and cache_key in graph.cache and graph.cache[cache_key] is not None:
            return graph.cache[cache_key]
        else:
            edge_index, edge_weight = gcn_norm_edge(graph.edge_index, graph.num_nodes, graph.edge_weight, improved=improved)
            if use_cache:
                graph.cache[cache_key] = edge_index, edge_weight
            return edge_index, edge_weight

    def call(self, inputs, training=None, mask=None):
        x, updated_edge_index, normed_edge_weight = inputs
        return gcn(x, updated_edge_index, normed_edge_weight, self.weight, self.bias, activation=self.acvitation)