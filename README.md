# tf_geometric
Efficient and Friendly Graph Neural Network Library for TensorFlow 1.x and 2.x.

Inspired by [rusty1s/pytorch_geometric](https://github.com/rusty1s/pytorch_geometric), we build a GNN library for TensorFlow.

## Support both TensorFlow 1.x and 2.x
This library is compatible with both TensorFlow 1.x and 2.x


## Efficient and Friendly

We use Message Passing mechanism to implement graph neural networks, which is way efficient than the dense matrix based implementations and more friendly than the sparse matrix based ones.

## A Map-Reduce Style Implementation

We provide map-reduce style APIs for programmers.

## OOP and Functional API

We provide both OOP and Functional API, with which you can make some cool things.


```python
# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tf_geometric as tfg
from tf_geometric.layers import GCN, MapReduceGNN
from tf_geometric.datasets.ppi import PPIDataset
import tensorflow as tf

# all graph data are in numpy format
train_data, valid_data, test_data = PPIDataset().load_data()

# we can convert them into tensorflow format
test_data = [graph.convert_data_to_tensor() for graph in test_data]


# ==================================== Basic OOP API ====================================
# OOP Style GCN
gcn_layer = GCN(units=20, activation=tf.nn.relu)

for graph in test_data:
    normed_edge_weight = GCN.create_normed_edge_weight(graph, use_cache=True)
    outputs = gcn_layer([graph.x, graph.edge_index, normed_edge_weight])
    print(outputs)


# ==================================== Basic Functional API ====================================
# Functional Style GCN
# Functional API is more flexible for advanced algorithms
# You can pass both data and parameters to functional APIs

dense_w = tf.Variable(tf.random.truncated_normal([test_data[0].num_features, 20]))
for graph in test_data:
    normed_edge_weight = tfg.nn.gcn_norm(graph.edge_index, graph.num_nodes)
    outputs = tfg.nn.gcn(graph.x, graph.edge_index, normed_edge_weight, dense_w, activation=tf.nn.relu)
    print(outputs)


# ==================================== Advanced OOP API ====================================
# All APIs are implemented with Map-Reduce Style
# This is a gcn without weight normalization and transformation.
# Create your own GNN Layer by subclassing the MapReduceGNN class
class NaiveGCN(MapReduceGNN):

    def map(self, repeated_x, neighbor_x, edge_weight=None):
        return tfg.nn.identity_mapper(repeated_x, neighbor_x, edge_weight)

    def reduce(self, neighbor_msg, node_index, num_nodes=None):
        return tfg.nn.sum_reducer(neighbor_msg, node_index, num_nodes)

    def update(self, x, reduced_neighbor_msg):
        return tfg.nn.sum_updater(x, reduced_neighbor_msg)


naive_gcn = NaiveGCN()

for graph in test_data:
    print(naive_gcn([graph.x, graph.edge_index, graph.edge_weight]))


# ==================================== Advanced Functional API ====================================
# All APIs are implemented with Map-Reduce Style
# This is a gcn without without weight normalization and transformation
# Just pass the mapper/reducer/updater functions to the Functional API

for graph in test_data:
    outputs = tfg.nn.aggregate_neighbors(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        mapper=tfg.nn.identity_mapper,
        reducer=tfg.nn.sum_reducer,
        updater=tfg.nn.sum_updater
    )
    print(outputs)

```