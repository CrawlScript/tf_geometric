# coding=utf-8

import tensorflow as tf
from tf_geometric.nn.conv.chebnet import normalization, chebnet
from tf_geometric.layers.kernel.map_reduce import MapReduceGNN


class chebNet(MapReduceGNN):
    """
    The chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    """


    def __init__(self,  units, K, lambda_max, activation=tf.nn.relu, use_bias=True, normalization_type='sys',
                 *args, **kwargs):
        """

        :param units: Positive integer, dimensionality of the output space.
        :param K: Chebyshev filter size (default: '3").
        :param lambda_max:
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param activation: Activation function to use.
        :param normalization_type: The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`)

        """
        super().__init__(*args, **kwargs)
        self.units = units

        assert K > 0
        assert lambda_max is not None
        assert normalization_type in [None, 'sym', 'rw'], 'Invalid normalization'

        self.K = K

        self.use_bias = use_bias

        self.kernel = None
        self.bias = None

        self.activation = activation
        self.normalization_type = normalization_type
        self.lambda_max = lambda_max

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[self.K, num_features, self.units], initializer="glorot_uniform")
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

        num_nodes = x.shape[0]
        norm_edge_index, norm_edge_weight = normalization(edge_index, num_nodes, edge_weight, self.lambda_max,
                                                          normalization_type=self.normalization_type)


        T0_x = x
        T1_x = x
        out = tf.matmul(T0_x, self.kernel[0])

        if self.K > 1:
            T1_x = chebnet(x, norm_edge_index, norm_edge_weight)
            out += tf.matmul(T1_x, self.kernel[1])

        for i in range(2,self.K):
            T2_x = chebnet(T1_x, norm_edge_index, norm_edge_weight) ##L^T_{k-1}(L^)
            T2_x = 2.0 * T2_x - T0_x
            out += tf.matmul(T2_x, self.kernel[i])

            T0_x, T1_x = T1_x, T2_x

        if self.use_bias:
            out += self.bias

        if self.activation is not None:
            out += self.activation(out)

        return out