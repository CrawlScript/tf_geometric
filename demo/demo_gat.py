# coding=utf-8
import tensorflow as tf
from tensorflow import keras
from tf_geometric.datasets.cora import CoraDataset
from tf_geometric.layers import GCN

graph, (train_index, valid_index, test_index) = CoraDataset().load_data()

num_classes = graph.y.shape[-1]

class GCNNetwork(keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gcn0 = GCN(16, activation=tf.nn.relu)
        self.gcn1 = GCN(num_classes)

    def call(self, inputs, training=None, mask=None):
        x, updated_edge_index, normed_edge_weight = inputs
        h = self.gcn0([x, updated_edge_index, normed_edge_weight])
        h = self.gcn1([h, updated_edge_index, normed_edge_weight])
        return h


updated_edge_index, normed_edge_weight = GCN.norm_edge(graph, use_cache=True)

gcn_network = GCNNetwork()

print(gcn_network([graph.x, updated_edge_index, normed_edge_weight]))


#
# print(graph)
# print(train_index)
# print(valid_index)
# print(test_index)