# coding=utf-8
import numpy as np
import tf_geometric as tfg
import tensorflow as tf

graph = tfg.Graph(
    x=np.random.randn(5, 20),  # 5 nodes, 20 features,
    edge_index=[[0, 0, 1, 3],
                [1, 2, 2, 1]]  # 4 undirected edges
)

print("Graph Desc: \n", graph)

graph.convert_edge_to_directed()  # pre-process edges
print("Processed Graph Desc: \n", graph)
print("Processed Edge Index:\n", graph.edge_index)

# Multi-head Graph Attention Network (GAT)
gat_layer = tfg.layers.GAT(units=4, num_heads=4, activation=tf.nn.relu)
output = gat_layer([graph.x, graph.edge_index])
print("Output of GAT: \n", output)