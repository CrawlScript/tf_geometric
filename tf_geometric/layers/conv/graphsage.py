# coding=utf-8
import tensorflow as tf

from tf_geometric.nn.conv.graphsage import graphSAGE as  SAGE
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

        ## mean aggregate
        self.mean_kernel = self.add_weight("kernel", shape=[num_features, self.units], initializer="glorot_uniform")
        self.mean_kernel_2 = self.add_weight("kernel", shape=[num_features, self.units], initializer="glorot_uniform")

        if self.use_bias:
            self.mean_kernel_bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")

        self.mean_hidden_kernel = self.add_weight("kernel", shape=[self.units * 2, self.units],
                                            initializer="glorot_uniform")
        self.mean_hidden_kernel_2 = self.add_weight("kernel", shape=[self.units * 2 , self.units],
                                             initializer="glorot_uniform")
        if self.use_bias:
            self.mean_hidden_bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")

        self.mean_out_kernel = self.add_weight("kernel", shape=[self.units * 2, self.units],
                                          initializer="glorot_uniform")
        self.mean_out_kernel_2 = self.add_weight("kernel", shape=[self.units * 2, self.units],
                                          initializer="glorot_uniform")

        if self.use_bias:
            self.mean_out_bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")

        ##gcn aggregate
        self.gcn_kernel = self.add_weight("kernel", shape=[num_features * 2, self.units], initializer="glorot_uniform")
        if self.use_bias:
            self.gcn_kernel_bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

        self.gcn_hidden_kernel = self.add_weight("kernel", shape=[self.units * 2, self.units], initializer="glorot_uniform")
        if self.use_bias:
            self.gcn_hidden_bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

        self.gcn_out_kernel = self.add_weight("kernel", shape=[self.units * 2, self.units * 2], initializer="glorot_uniform")
        if self.use_bias:
            self.gcn_out_bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")

        ##mean_pooling and max_pooling
        self.first__mlp_kernel = self.add_weight("kernel", shape=[num_features, self.units], initializer="glorot_uniform")
        if self.use_bias:
            self.first_mlp_bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

        self.first_pooling_kernel = self.add_weight("kernel", shape=[self.units, self.units],
                                             initializer="glorot_uniform")
        self.first_pooling_kernel_2 = self.add_weight("kernel", shape=[num_features, self.units],
                                              initializer="glorot_uniform")
        if self.use_bias:
            self.first_pooling_bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")


        self.hidden_mlp_kernel = self.add_weight("kernel", shape=[self.units * 2, self.units],
                                                     initializer="glorot_uniform")
        self.hidden_pooling_kernel = self.add_weight("kernel", shape=[self.units, self.units],
                                                     initializer="glorot_uniform")
        self.hidden_pooling_kernel_2 = self.add_weight("kernel", shape=[self.units * 2, self.units],
                                             initializer="glorot_uniform")
        if self.use_bias:
            self.hidden_mlp_bias = self.add_weight("bias", shape=[self.units], initializer="zeros")
        if self.use_bias:
            self.hidden_pooling_bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")



        self.out_mlp_kernel = self.add_weight("kernel", shape=[self.units * 2, self.units],
                                        initializer="glorot_uniform")
        self.out_pooling_kernel = self.add_weight("kernel", shape=[self.units, self.units],
                                                  initializer="glorot_uniform")
        self.out_pooling_kernel_2 = self.add_weight("kernel", shape=[self.units * 2, self.units],
                                                  initializer="glorot_uniform")

        if self.use_bias:
            self.out_mlp_bias = self.add_weight("bias", shape=[self.units], initializer="zeros")
        if self.use_bias:
            self.out_pooling_bias = self.add_weight("bias", shape=[self.units * 2], initializer="zeros")


        self.classifier = self.add_weight("kernel", shape=[self.units * 2, self.num_classes],
                                        initializer="glorot_uniform")
        if self.use_bias:
            self.classifier_bias = self.add_weight("bias", shape=[self.num_classes], initializer="zeros")

        self.dropout = keras.layers.Dropout(self.drop_rate)



    def call(self, inputs, num_layers=3, training=False):

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = tf.ones(edge_index.shape[-1],dtype=tf.int32)

        if self.aggregate_type == 'mean_pooling' or self.aggregate_type == 'max_pooling':
            kernel = self.first__mlp_kernel
            bias = self.first_mlp_bias
            kernel_2 = self.first_pooling_kernel
            kernel_3 = self.first_pooling_kernel_2
            bias_2 = self.first_pooling_bias

            hidden_kernel = self.hidden_mlp_kernel
            hidden_bias = self.hidden_mlp_bias
            hidden_kernel_2 = self.hidden_pooling_kernel
            hidden_kernel_3 = self.hidden_pooling_kernel_2
            hidden_bias_2 = self.hidden_pooling_bias

            out_kernel = self.out_mlp_kernel
            out_bias = self.out_mlp_bias
            out_kernel_2 = self.out_pooling_kernel
            out_kernel_3 = self.out_pooling_kernel_2
            out_bias_2 = self.out_pooling_bias


        elif self.aggregate_type == 'mean':
            kernel = self.mean_kernel
            kernel_2 = self.mean_kernel_2
            kernel_3 = None
            bias = self.mean_kernel_bias
            bias_2 = None

            hidden_kernel = self.mean_hidden_kernel
            hidden_bias = self.mean_hidden_bias
            hidden_kernel_2 = self.mean_hidden_kernel_2
            hidden_bias_2 = None
            hidden_kernel_3 = None

            out_kernel = self.mean_out_kernel
            out_bias = self.mean_out_bias
            out_kernel_2 = self.mean_out_kernel_2
            out_bias_2 = None
            out_kernel_3 = None

        elif self.aggregate_type == 'gcn':
            kernel = self.gcn_kernel
            kernel_2 = None
            kernel_3 = None
            bias = self.gcn_kernel_bias
            bias_2 = None

            hidden_kernel = self.gcn_hidden_kernel
            hidden_bias = self.gcn_hidden_bias
            hidden_kernel_2 = None
            hidden_bias_2 = None
            hidden_kernel_3 = None

            out_kernel = self.gcn_out_kernel
            out_bias = self.gcn_out_bias
            out_kernel_2 = None
            out_bias_2 = None
            out_kernel_3 = None



        h = SAGE(x, edge_index[0], edge_weight[0], kernel, kernel_2, kernel_3, bias, bias_2,
                 aggregate_type=self.aggregate_type, activation=self.acvitation)


        for i in range(1,num_layers-1):
            h = SAGE(h, edge_index[i], edge_weight[i], hidden_kernel, hidden_kernel_2, hidden_kernel_3,hidden_bias,
                     hidden_bias_2,  aggregate_type=self.aggregate_type, activation=self.acvitation)

            if i != num_layers-1:
                h = self.dropout(h,training=training)

        h = SAGE(h, edge_index[-1], edge_weight[-1], out_kernel, out_kernel_2, out_kernel_3, out_bias, out_bias_2,
                 aggregate_type=self.aggregate_type, activation=self.acvitation)


        out =  h @ self.classifier
        out += self.classifier_bias
        # out = tf.nn.relu(out)
        return out

