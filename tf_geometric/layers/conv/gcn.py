# coding=utf-8

import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from tf_geometric import Graph
from tf_geometric.nn.conv.gcn import gcn_norm_edge, gcn
from tf_geometric.layers.kernel.map_reduce import MapReduceGNN


class GCN(MapReduceGNN):

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units], initializer="glorot_uniform")
        self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

    def __init__(self, units, activation=None, improved=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units

        self.acvitation = activation
        self.kernel = None
        self.bias = None

        self.improved = improved

    def call(self, inputs, training=None, mask=None, cache=None):
        """

        :param inputs:
        :param training:
        :param mask:
        :param graph: If graph is provided, it is used for caching normed edge info
        :return:
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return gcn(x, edge_index, edge_weight, self.kernel, self.bias,
                   activation=self.acvitation, improved=self.improved, cache=cache)