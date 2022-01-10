# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.conv.gat import gat


class GAT(tf.keras.Model):

    def __init__(self, units,
                 attention_units=None,
                 activation=None,
                 use_bias=True,
                 num_heads=1,
                 split_value_heads=True,
                 query_activation=tf.nn.relu,
                 key_activation=tf.nn.relu,
                 edge_drop_rate=0.0,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 *args, **kwargs):
        """

        :param units: Positive integer, dimensionality of the output space.
        :param attention_units: Positive integer, dimensionality of the output space for Q and K in attention.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param num_heads: Number of attention heads.
        :param split_value_heads: Boolean. If true, split V as value attention heads, and then concatenate them as output.
            Else, num_heads replicas of V are used as value attention heads, and the mean of them are used as output.
        :param query_activation: Activation function for Q in attention.
        :param key_activation: Activation function for K in attention.
        :param edge_drop_rate: Dropout rate of attention weights.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        """
        super().__init__(*args, **kwargs)
        self.units = units
        self.attention_units = units if attention_units is None else attention_units
        self.edge_drop_rate = edge_drop_rate

        self.query_kernel = None
        self.query_bias = None
        self.query_activation = query_activation

        self.key_kernel = None
        self.key_bias = None
        self.key_activation = key_activation

        self.kernel = None
        self.bias = None

        self.activation = activation
        self.use_bias = use_bias
        self.num_heads = num_heads
        self.split_value_heads = split_value_heads

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.query_kernel = self.add_weight("query_kernel", shape=[num_features, self.attention_units],
                                            initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.query_bias = self.add_weight("query_bias", shape=[self.attention_units],
                                          initializer="zeros", regularizer=self.bias_regularizer)

        self.key_kernel = self.add_weight("key_kernel", shape=[num_features, self.attention_units],
                                          initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.key_bias = self.add_weight("key_bias", shape=[self.attention_units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index] or [x, edge_index, edge_weight].
            Note that the edge_weight will not be used.
        :return: Updated node features (x), shape: [num_nodes, units]
        """
        x, edge_index = inputs[0], inputs[1]

        return gat(x, edge_index,
                   self.query_kernel, self.query_bias, self.query_activation,
                   self.key_kernel, self.key_bias, self.key_activation,
                   self.kernel, self.bias, self.activation,
                   num_heads=self.num_heads,
                   split_value_heads=self.split_value_heads,
                   edge_drop_rate=self.edge_drop_rate,
                   training=training)
