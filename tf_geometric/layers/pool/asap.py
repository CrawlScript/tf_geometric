# coding=utf-8


import tensorflow as tf
from tf_geometric.nn.pool.asap import asap


class ASAP(tf.keras.Model):
    """
    OOP API for ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representation
    """

    def __init__(self, k=None, ratio=None, drop_rate=0.0,
                 attention_units=None,
                 le_conv_activation=tf.nn.sigmoid,
                 le_conv_use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None,
                 *args, **kwargs):
        """
        ASAPool
        :param attention_units: Positive integer, dimensionality for attention.
        """
        super().__init__(*args, **kwargs)

        self.attention_units = attention_units

        self.attention_gcn_kernel = None
        self.attention_gcn_bias = None
        self.attention_query_kernel = None
        self.attention_query_bias = None
        self.attention_score_kernel = None
        self.attention_score_bias = None

        self.le_conv_self_kernel = None
        self.le_conv_self_bias = None
        self.le_conv_aggr_self_kernel = None
        self.le_conv_aggr_self_bias = None
        self.le_conv_aggr_neighbor_kernel = None

        self.k = k
        self.ratio = ratio
        self.drop_rate = drop_rate
        self.le_conv_activation = le_conv_activation

        self.le_conv_use_bias = le_conv_use_bias

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        if self.attention_units is None:
            self.attention_units = num_features



        self.attention_gcn_kernel = self.add_weight("attention_gcn_kernel", shape=[num_features, self.attention_units],
                                                    initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.attention_gcn_bias = self.add_weight("attention_gcn_bias", shape=[self.attention_units],
                                        initializer="zeros", regularizer=self.bias_regularizer)
        self.attention_query_kernel = self.add_weight("attention_query_kernel", shape=[self.attention_units, self.attention_units],
                                                    initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.attention_query_bias = self.add_weight("attention_query_bias", shape=[self.attention_units],
                                        initializer="zeros", regularizer=self.bias_regularizer)
        self.attention_score_kernel = self.add_weight("attention_score_kernel", shape=[self.attention_units * 2, 1],
                                                    initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.attention_score_bias = self.add_weight("attention_score_bias", shape=[1],
                                        initializer="zeros", regularizer=self.bias_regularizer)

        self.le_conv_self_kernel = self.add_weight("le_conv_self_kernel", shape=[self.attention_units, 1],
                                                   initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        if self.le_conv_use_bias:
            self.le_conv_self_bias = self.add_weight("le_conv_self_bias", shape=[1],
                                                     initializer="zeros", regularizer=self.bias_regularizer)

        self.le_conv_aggr_self_kernel = self.add_weight("le_conv_aggr_self_kernel", shape=[self.attention_units, 1],
                                                        initializer="glorot_uniform", regularizer=self.kernel_regularizer)

        if self.le_conv_use_bias:
            self.le_conv_aggr_self_bias = self.add_weight("le_conv_aggr_self_bias", shape=[1],
                                                      initializer="zeros", regularizer=self.bias_regularizer)

        self.le_conv_aggr_neighbor_kernel = self.add_weight("le_conv_aggr_neighbor_kernel", shape=[self.attention_units, 1],
                                                            initializer="glorot_uniform", regularizer=self.kernel_regularizer)


    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight, node_graph_index]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """
        x, edge_index, edge_weight, node_graph_index = inputs

        return asap(x, edge_index, edge_weight, node_graph_index,
                    self.attention_gcn_kernel, self.attention_gcn_bias,
                    self.attention_query_kernel, self.attention_query_bias,
                    self.attention_score_kernel, self.attention_score_bias,
                    self.le_conv_self_kernel, self.le_conv_self_bias,
                    self.le_conv_aggr_self_kernel, self.le_conv_aggr_self_bias,
                    self.le_conv_aggr_neighbor_kernel, None,
                    k=self.k, ratio=self.ratio, le_conv_activation=self.le_conv_activation,
                    drop_rate=self.drop_rate, training=training, cache=cache)
