# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.conv.graphSAGE import graphSAGE as  SAGE
from tf_geometric.layers.kernel.map_reduce import MapReduceGNN
from tensorflow.python import keras


class GraphSAGE(MapReduceGNN):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
       Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
       """
    def __init__(self,hidden_units, num_classes, aggregate_type='mean', activation=tf.nn.relu, use_bias=True,
                drop_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = hidden_units
        self.num_classes = num_classes

        self.aggregate_type = aggregate_type
        self.acvitation = activation
        self.use_bias = use_bias

        self.kernel = None
        self.bias = None

        self.drop_rate = drop_rate

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        ## mean aggregate and gcn aggregate
        self.kernel = self.add_weight("kernel", shape=[num_features*2, self.units], initializer="glorot_uniform")
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

        self.hidden_kernel = self.add_weight("kernel", shape=[self.units * 2, self.units],
                                            initializer="glorot_uniform")
        if self.use_bias:
            self.hidden_bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

        self.out_kernel = self.add_weight("kernel", shape=[self.units * 2, self.num_classes],
                                          initializer="glorot_uniform")
        if self.use_bias:
            self.out_bias = self.add_weight("bias", shape=[self.num_classes], initializer="zeros")



        ##mean_pooling and max_pooling
        self.pooling_kernel = self.add_weight("kernel", shape=[num_features, self.units], initializer="glorot_uniform")
        if self.use_bias:
            self.pooling_bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

        self.kernel_2 = self.add_weight("kernel", shape=[self.units + num_features, self.units],
                                             initializer="glorot_uniform")
        if self.use_bias:
            self.bias_2 = self.add_weight("bias", shape=[self.units], initializer="zeros")

        self.hidden_pooling_kernel = self.add_weight("kernel", shape=[self.units, self.units],
                                             initializer="glorot_uniform")
        if self.use_bias:
            self.hidden_pooling_bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

        self.kernel_3 = self.add_weight("kernel", shape=[self.units * 2, self.units],
                                          initializer="glorot_uniform")
        if self.use_bias:
            self.bias_3 = self.add_weight("bias", shape=[self.units], initializer="zeros")

        self.out_pooling_kernel = self.add_weight("kernel", shape=[self.units, self.units],
                                        initializer="glorot_uniform")
        if self.use_bias:
            self.out_pooling_bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

        self.kernel_4 = self.add_weight("kernel", shape=[self.units * 2, self.num_classes],
                                        initializer="glorot_uniform")
        if self.use_bias:
            self.bias_4 = self.add_weight("bias", shape=[self.num_classes], initializer="zeros")


        self.dropout = keras.layers.Dropout(self.drop_rate)



    def call(self, inputs, num_layers=3, training=None):

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = tf.ones(edge_index.shape[-1],dtype=tf.int32)

        if self.aggregate_type == 'mean_pooling' or self.aggregate_type == 'max_pooling':
            kernel = self.pooling_kernel
            bias = self.pooling_bias
            kernel_2 = self.kernel_2
            bias_2 = self.bias_2
            hidden_kernel = self.hidden_pooling_kernel
            hidden_bias = self.hidden_pooling_bias
            hidden_kernel_2 = self.kernel_3
            hidden_bias_2 = self.bias_3
            out_kernel = self.out_pooling_kernel
            out_bias = self.out_pooling_bias
            out_kernel_2 = self.kernel_4
            out_bias_2 = self.bias_4


        else:
            kernel = self.kernel
            bias = self.bias
            kernel_2 = None
            bias_2 = None
            hidden_kernel = self.hidden_kernel
            hidden_bias = self.hidden_bias
            hidden_kernel_2 = None
            hidden_bias_2 = None
            out_kernel = self.out_kernel
            out_bias = self.out_bias
            out_kernel_2 = None
            out_bias_2 = None



        h = SAGE(x, edge_index[0], edge_weight[0], kernel, kernel_2, bias, bias_2,
                 aggregate_type=self.aggregate_type, activation=self.acvitation)

        for i in range(1,num_layers-1):
            h = SAGE(h, edge_index[i], edge_weight[i], hidden_kernel, hidden_kernel_2, hidden_bias,
                     hidden_bias_2,  aggregate_type=self.aggregate_type, activation=self.acvitation)

            if i != num_layers-1:
                h = self.dropout(h,training=training)

        out = SAGE(h, edge_index[-1], edge_weight[-1], out_kernel, out_kernel_2, out_bias, out_bias_2,
                   aggregate_type=self.aggregate_type, activation=self.acvitation)


        return out

