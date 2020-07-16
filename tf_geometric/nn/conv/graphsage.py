import tensorflow as tf
from tensorflow import keras
from tf_geometric.nn.kernel.segment import segment_count, segment_op_with_pad
from tf_geometric.nn.conv.gcn import gcn_mapper
from tf_geometric.utils.graph_utils import add_self_loop_edge

def mean_reducer(neighbor_msg, node_index, num_nodes=None):
    return tf.math.unsorted_segment_mean(neighbor_msg, node_index, num_segments=num_nodes)

def max_reducer(neighbor_msg, node_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = tf.reduce_max(node_index) + 1
    # max_x = tf.math.unsorted_segment_max(x, node_graph_index, num_segments=num_graphs)
    max_x = segment_op_with_pad(tf.math.segment_max, neighbor_msg, node_index, num_segments=num_nodes)
    return max_x


class GraphSAGE(keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def reduce(self, x, edge_index, kernel_1, kernel_2, edge_weight, bias,
               activation, normalize):
        pass
    def build(self, input_shape):
        pass
    def call(self, inputs, training=None, mask=None):
        pass



class Mean_Aggregator(GraphSAGE):
    def __init__(self, units, activation=tf.nn.relu, use_bias=True, dropout_rate=None,
                 normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.normalize = normalize

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.neighs_kernel = self.add_weight("neighs_kernel", shape=[num_features, self.units],
                                               initializer="glorot_uniform")
        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, self.units],
                                           initializer="glorot_uniform")

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """
        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        row, col = edge_index
        repeated_x = tf.gather(x, row)
        neighbor_x = tf.gather(x, col)

        if edge_weight is not None:
            neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

        neighbor_reduced_msg = mean_reducer(neighbor_x, row, num_nodes=len(x))

        neighbor_msg = neighbor_reduced_msg @ self.neighs_kernel
        x = x @ self.self_kernel
        h = tf.concat([neighbor_msg, x], axis=1)

        if self.use_bias is not None:
            h += self.bias

        if self.activation is not None:
            h = self.activation(h)

        if self.normalize:
            h = tf.nn.l2_normalize(h, axis=-1)

        return h

class GCN_Aggregator(GraphSAGE):
    def __init__(self, units, activation=tf.nn.relu, use_bias=True, dropout_rate= None,
                  normalize=False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.normalize = normalize

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[num_features * 2, self.units],
                                               initializer="glorot_uniform")

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        if edge_weight is not None:
            edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)

        edge_index, edge_weight = add_self_loop_edge(edge_index, x.shape[0], edge_weight=edge_weight, fill_weight=2.0)

        row, col = edge_index
        repeated_x = tf.gather(x, row)
        neighbor_x = tf.gather(x, col)

        neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

        reduced_msg = mean_reducer(neighbor_x, row, num_nodes=len(x))
        updated_msg = tf.concat([reduced_msg, x], axis=1)

        h = updated_msg @ self.kernel
        if self.use_bias is not None:
            h += self.bias

        if self.activation is not None:
            h = self.activation(h)

        if self.normalize:
            h = tf.nn.l2_normalize(h, axis=-1)

        return h


class MeanPooling_Aggregator(GraphSAGE):
    def __init__(self, units, activation=tf.nn.relu, use_bias=True, dropout_rate=0.5,
                 normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.normalize = normalize

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.mlp_kernel = self.add_weight("mlp_kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform")
        if self.use_bias:
            self.mlp_bias = self.add_weight("mlp_bias", shape=[self.units], initializer="zeros")
        # self.mlp_kernel = keras.layers.Dense(self.units, input_dim=2, use_bias=True, kernel_regularizer= tf.nn.l2_normalize, activation=tf.nn.relu)

        self.neighs_kernel = self.add_weight("neighs_kernel", shape=[self.units, self.units],
                                      initializer="glorot_uniform")
        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform")

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")

        self.dropout = keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        if edge_weight is not None:
            edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)


        row, col = edge_index
        repeated_x = tf.gather(x, row)
        neighbor_x = tf.gather(x, col)

        neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

        neighbor_x = self.dropout(neighbor_x)
        h = neighbor_x @ self.mlp_kernel
        if self.use_bias:
            h += self.mlp_bias

        if self.activation is not None:
            h = self.activation(h)

        reduced_h = mean_reducer(h, row, num_nodes=len(x))

        from_neighs = reduced_h @ self.neighs_kernel
        from_x = x @ self.self_kernel

        output = tf.concat([from_neighs, from_x], axis=1)
        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        if self.normalize:
            output = tf.nn.l2_normalize(output, axis=-1)

        return output

class MaxPooling_Aggregator(GraphSAGE):
    def __init__(self, units, activation=tf.nn.relu, use_bias=True, dropout_rate=0.5,
                 normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.normalize = normalize

    def build(self, input_shape):
        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.mlp_kernel = self.add_weight("mlp_kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform")
        if self.use_bias:
            self.mlp_bias = self.add_weight("mlp_bias", shape=[self.units], initializer="zeros")
        self.neighs_kernel = self.add_weight("neighs_kernel", shape=[self.units, self.units],
                                      initializer="glorot_uniform")
        self.self_kernel = self.add_weight("self_kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform")

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")

        self.dropout = keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        if edge_weight is not None:
            edge_weight = tf.ones([edge_index.shape[1]], dtype=tf.float32)


        row, col = edge_index
        repeated_x = tf.gather(x, row)
        neighbor_x = tf.gather(x, col)

        neighbor_x = gcn_mapper(repeated_x, neighbor_x, edge_weight=edge_weight)

        neighbor_x = self.dropout(neighbor_x)
        h = neighbor_x @ self.mlp_kernel
        if self.use_bias:
            h += self.mlp_bias

        if self.activation is not None:
            h = self.activation(h)

        reduced_h = max_reducer(h, row, num_nodes=len(x))

        from_neighs = reduced_h @ self.neighs_kernel
        from_x = x @ self.self_kernel

        output = tf.concat([from_neighs, from_x], axis=1)
        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        if self.normalize:
            output = tf.nn.l2_normalize(output, axis=-1)

        return output
