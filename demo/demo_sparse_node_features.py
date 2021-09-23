# coding=utf-8
import os
# Use GPU0
from tf_geometric.utils import tf_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import tf_geometric as tfg


num_nodes = 5
edge_index = [
    [0, 0, 1, 2, 3, 3, 4],
    [1, 2, 4, 3, 1, 4, 2]
]

# Sparse node features
# tf.sparse.eye creates a two-dimensional sparse tensor with ones along the diagonal
# x is the one-hot encoding of node ids (from 0 to num_nodes - 1) in the form of a sparse matrix
# This is usually used for feature-less cases, such as recommendation systems.
x = tf.sparse.eye(num_nodes)
print("Sparse (One-hot) Node Features: ")
print(tf.sparse.to_dense(x))

# tf.sparse.SparseTensor can be used as node features (x)
graph = tfg.Graph(x, edge_index).convert_edge_to_directed()
print("\nConstructed Graph:")
print(graph)

# create a one-layer GNN model
model = tfg.layers.GCN(4)
# model = tfg.layers.SGC(4, k=3)
# model = tfg.layers.ChebyNet(4, k=4)
# model = tfg.layers.TAGCN(4, k=4)

# predict with the GCN model
@tf_utils.function
def forward(graph):
    return model([graph.x, graph.edge_index])

logits = forward(graph)
print("\nModel Output:")
print(logits)

# tfg.Graph objects with sparse node features can also be combined into a tfg.BatchGraph object
batch_graph = tfg.BatchGraph.from_graphs([graph, graph])
print("\nCombined Batch Graph")
print(batch_graph)
