# tf_geometric

Efficient and Friendly Graph Neural Network Library for TensorFlow 1.x and 2.x.

Inspired by __rusty1s/pytorch_geometric__, we build a GNN library for TensorFlow.

## HomePage

[https://github.com/CrawlScript/tf_geometric](https://github.com/CrawlScript/tf_geometric)





## Efficient and Friendly

We use Message Passing mechanism to implement graph neural networks, which is way efficient than the dense matrix based implementations and more friendly than the sparse matrix based ones.
In addition, we provide easy and elegant APIs for complex GNN operations.
The following example constructs a graph and applies a Multi-head Graph Attention Network (GAT) on it:
```python
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
```

Output:
```
Graph Desc:
 Graph Shape: x => (5, 20)	edge_index => (2, 4)	y => None

Processed Graph Desc:
 Graph Shape: x => (5, 20)	edge_index => (2, 8)	y => None

Processed Edge Index:
 [[0 0 1 1 1 2 2 3]
 [1 2 0 2 3 0 1 1]]

Output of GAT:
 tf.Tensor(
[[0.22443159 0.         0.58263206 0.32468423]
 [0.29810357 0.         0.19403605 0.35630274]
 [0.18071976 0.         0.58263206 0.32468423]
 [0.36123228 0.         0.88897204 0.450244  ]
 [0.         0.         0.8013462  0.        ]], shape=(5, 4), dtype=float32)
```




## OOP and Functional API

We provide both OOP and Functional API, with which you can make some cool things.
You can learn more about them in the document.



## DEMO

### Node Classification

+ [Graph Convolutional Network (GCN)](demo/demo_gcn.py)
+ [Multi-head Graph Attention Network (GAT)](demo/demo_gat.py)

### Link Prediction

+ [MeanPooling](demo/demo_mean_pool.py)
+ [SAGPooling](demo/demo_sag_pool_h.py)


### Graph Classification

+ [Graph Auto-Encoder (GAE)](demo/demo_gae.py)


## Installation

Requirements:
+ Operation System: Windows / Linux / Mac OS
+ Python: version >= 3.5
+ Python Packages:
    + tensorflow/tensorflow-gpu: >= 1.14.0 or >= 2.0.0b1
    + numpy >= 1.17.4
    + networkx >= 2.1
    + scipy >= 1.1.0


Use one of the following commands below:
```bash
pip install -U tf_geometric # this will not install the tensorflow/tensorflow-gpu package

pip install -U tf_geometric[tf1-cpu] # this will install TensorFlow 1.x CPU version

pip install -U tf_geometric[tf1-gpu] # this will install TensorFlow 1.x GPU version

pip install -U tf_geometric[tf2-cpu] # this will install TensorFlow 2.x CPU version

pip install -U tf_geometric[tf2-gpu] # this will install TensorFlow 2.x GPU version
```