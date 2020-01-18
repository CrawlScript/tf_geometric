# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.conv.gat import gat
from tf_geometric.layers.kernel.map_reduce import MapReduceGNN


class GAT(MapReduceGNN):

    def __init__(self, units, activation=None, query_activation=tf.nn.relu, key_activation=tf.nn.relu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

        self.query_kernel = None
        self.query_bias = None
        self.query_activation = query_activation

        self.key_kernel = None
        self.key_bias = None
        self.key_activation = key_activation

        self.kernel = None
        self.bias = None

        self.acvitation = activation

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        query_units = 1
        key_units = 1

        self.query_kernel = self.add_weight("query_kernel", shape=[num_features, query_units], initializer="glorot_uniform")
        self.query_bias = self.add_weight("query_bias", shape=[query_units], initializer="zeros")

        self.key_kernel = self.add_weight("key_kernel", shape=[num_features, key_units], initializer="glorot_uniform")
        self.key_bias = self.add_weight("key_bias", shape=[key_units], initializer="zeros")

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units], initializer="glorot_uniform")
        self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

    def call(self, inputs, training=None, mask=None):
        x, edge_index = inputs

        return gat(x, edge_index,
                   self.query_kernel, self.query_bias, self.query_activation,
                   self.key_kernel, self.key_bias, self.key_activation,
                   self.kernel, self.bias, self.acvitation)