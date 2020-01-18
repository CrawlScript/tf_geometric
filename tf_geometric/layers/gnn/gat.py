# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.gnn.gat import gat
from tf_geometric.layers.kernel.map_reduce import MapReduceGNN


class GAT(MapReduceGNN):

    def __init__(self, units, activation=None, use_cache=True, query_activation=tf.nn.relu, key_activation=tf.nn.relu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.use_cache = use_cache

        self.query_weight = None
        self.query_bias = None
        self.query_activation = query_activation

        self.key_weight = None
        self.key_bias = None
        self.key_activation = key_activation

        self.weight = None
        self.bias = None

        self.acvitation = activation

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.query_weight = tf.Variable(tf.truncated_normal([num_features, self.units]))
        self.query_bias = tf.Variable(tf.zeros([self.units]))

        self.key_weight = tf.Variable(tf.truncated_normal([num_features, self.units]))
        self.key_bias = tf.Variable(tf.zeros([self.units]))

        self.weight = tf.Variable(tf.truncated_normal([num_features, self.units]))
        self.bias = tf.Variable(tf.zeros([self.units]))

    def call(self, inputs, training=None, mask=None):
        if len(inputs) == 2:
            x, edge_index = inputs
            edge_weight = None
        else:
            x, edge_index, edge_weight = inputs

        return gat(x, edge_index, edge_weight,
                   self.query_weight, self.query_bias, self.query_activation,
                   self.key_weight, self.key_bias, self.key_activation,
                   self.weight, self.bias, self.acvitation)