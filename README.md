<p align="center">
<img src="TF_GEOMETRIC_LOGO.png" width="400"/>
</p>


# tf_geometric

Efficient and Friendly Graph Neural Network Library for TensorFlow 1.x and 2.x.

Inspired by __rusty1s/pytorch_geometric__, we build a GNN library for TensorFlow.


## Homepage and Documentation

+ Homepage: [https://github.com/CrawlScript/tf_geometric](https://github.com/CrawlScript/tf_geometric)
+ Documentation: [https://tf-geometric.readthedocs.io](https://tf-geometric.readthedocs.io) ([中文版](https://tf-geometric.readthedocs.io/en/latest/index_cn.html))
+ Paper: [Efficient Graph Deep Learning in TensorFlow with tf_geometric](https://arxiv.org/abs/2101.11552)


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

graph = graph.to_directed()  # pre-process edges
print("Processed Graph Desc: \n", graph)
print("Processed Edge Index:\n", graph.edge_index)

# Multi-head Graph Attention Network (GAT)
gat_layer = tfg.layers.GAT(units=4, num_heads=4, activation=tf.nn.relu)
output = gat_layer([graph.x, graph.edge_index])
print("Output of GAT: \n", output)
```

Output:
```html
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


## DEMO

We recommend you to get started with some demo.


### Node Classification

+ [Graph Convolutional Network (GCN)](demo/demo_gcn.py)
+ [Multi-head Graph Attention Network (GAT)](demo/demo_gat.py)
+ [Approximate Personalized Propagation of Neural Predictions (APPNP)](demo/demo_appnp.py)
+ [Inductive Representation Learning on Large Graphs (GraphSAGE)](demo/demo_graph_sage_func.py)
+ [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (ChebyNet)](demo/demo_chebynet.py)
+ [Simple Graph Convolution (SGC)](demo/demo_sgc.py)
+ [Topology Adaptive Graph Convolutional Network (TAGCN)](demo/demo_tagcn.py)
+ [Deep Graph Infomax (DGI)](demo/demo_dgi.py)
+ [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification (DropEdge)](demo/demo_drop_edge_gcn.py)
+ [Graph Convolutional Networks for Text Classification (TextGCN)](https://github.com/CrawlScript/TensorFlow-TextGCN)
+ [Simple Spectral Graph Convolution (SSGC/S^2GC)](demo/demo_ssgc.py)


### Graph Classification

+ [MeanPooling](demo/demo_mean_pool.py)
+ [Graph Isomorphism Network (GIN)](demo/demo_gin.py)
+ [Self-Attention Graph Pooling (SAGPooling)](demo/demo_sag_pool_h.py)
+ [Hierarchical Graph Representation Learning with Differentiable Pooling (DiffPool)](demo/demo_diff_pool.py)
+ [Order Matters: Sequence to Sequence for Sets (Set2Set)](demo/demo_set2set.py)
+ [ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations (ASAP)](demo/demo_asap.py)
+ [An End-to-End Deep Learning Architecture for Graph Classification (SortPool)](demo/demo_sort_pool.py)
+ [Spectral Clustering with Graph Neural Networks for Graph Pooling (MinCutPool)](demo/demo_min_cut_pool.py)



### Link Prediction

+ [Graph Auto-Encoder (GAE)](demo/demo_gae.py)



### Save and Load Models

+ [Save and Load Models](demo/demo_save_and_load_model.py)
+ [Save and Load Models with tf.train.Checkpoint](demo/demo_checkpoint.py)


### Distributed Training

+ [Distributed GCN for Node Classification](demo/demo_distributed_gcn.py)
+ [Distributed MeanPooling for Graph Classification](demo/demo_distributed_mean_pool.py)


### Sparse

+ [Sparse Node Features](demo/demo_sparse_node_features.py)



## Installation

Requirements:
+ Operation System: Windows / Linux / Mac OS
+ Python: version >= 3.5 and version != 3.6
+ Python Packages:
    + tensorflow/tensorflow-gpu: >= 1.15.0 or >= 2.3.0
    + tf_sparse
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

## OOP and Functional API

We provide both OOP and Functional API, with which you can make some cool things.

```python
# coding=utf-8
import os
# Enable GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tf_geometric as tfg
import tensorflow as tf
import numpy as np

# ==================================== Graph Data Structure ====================================
# In tf_geometric, the data of a graph can be represented by either a collections of
# tensors (numpy.ndarray or tf.Tensor) or a tfg.Graph object.
# A graph usually consists of x(node features), edge_index and edge_weight(optional)

# Node Features => (num_nodes, num_features)
x = np.random.randn(5, 20).astype(np.float32)  # 5 nodes, 20 features

# Edge Index => (2, num_edges)
# Each column of edge_index (u, v) represents an directed edge from u to v.
# Note that it does not cover the edge from v to u. You should provide (v, u) to cover it.
# This is not convenient for users.
# Thus, we allow users to provide edge_index in undirected form and convert it later.
# That is, we can only provide (u, v) and convert it to (u, v) and (v, u) with `convert_edge_to_directed` method.
edge_index = np.array([
    [0, 0, 1, 3],
    [1, 2, 2, 1]
])

# Edge Weight => (num_edges)
edge_weight = np.array([0.9, 0.8, 0.1, 0.2]).astype(np.float32)


# Usually, we use a graph object to manager these information
# edge_weight is optional, we can set it to None if you don't need it
# Using 'to_directed' to obtain a graph with directed edges such that we can use it as the input of GCN
graph = tfg.Graph(x=x, edge_index=edge_index, edge_weight=edge_weight).to_directed()


# Define a Graph Convolutional Layer (GCN)
gcn_layer = tfg.layers.GCN(4, activation=tf.nn.relu)
# Perform GCN on the graph
h = gcn_layer([graph.x, graph.edge_index, graph.edge_weight])
print("Node Representations (GCN on a Graph): \n", h)

for _ in range(10):
    # Using Graph.cache can avoid recomputation of GCN's normalized adjacency matrix,
    # which can dramatically improve the efficiency of GCN.
    h = gcn_layer([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)


# For algorithms that deal with batches of graphs, we can pack a batch of graph into a BatchGraph object
# Batch graph wrap a batch of graphs into a single graph, where each nodes has an unique index and a graph index.
# The node_graph_index is the index of the corresponding graph for each node in the batch.
# The edge_graph_index is the index of the corresponding edge for each node in the batch.
batch_graph = tfg.BatchGraph.from_graphs([graph, graph, graph, graph, graph])

# We can reversely split a BatchGraph object into Graphs objects
graphs = batch_graph.to_graphs()

# Define a Graph Convolutional Layer (GCN)
batch_gcn_layer = tfg.layers.GCN(4, activation=tf.nn.relu)
# Perform GCN on the BatchGraph
batch_h = gcn_layer([batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight])
print("Node Representations (GCN on a BatchGraph): \n", batch_h)

# Graph Pooling algorithms often rely on such batch data structure
# Most of them accept a BatchGraph's data as input and output a feature vector for each graph in the batch
graph_h = tfg.nn.mean_pool(batch_h, batch_graph.node_graph_index, num_graphs=batch_graph.num_graphs)
print("Graph Representations (Mean Pooling on a BatchGraph): \n", batch_h)


# Define a Graph Convolutional Layer (GCN) for scoring each node
gcn_score_layer = tfg.layers.GCN(1)
# We provide some advanced graph pooling operations such as topk_pool
node_score = gcn_score_layer([batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight])
node_score = tf.reshape(node_score, [-1])
print("Score of Each Node: \n", node_score)
topk_node_index = tfg.nn.topk_pool(batch_graph.node_graph_index, node_score, ratio=0.6)
print("Top-k Node Index (Top-k Pooling): \n", topk_node_index)




# ==================================== Built-in Datasets ====================================
# all graph data are in numpy format

# Cora Dataset
graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()

# PPI Dataset
train_data, valid_data, test_data = tfg.datasets.PPIDataset().load_data()

# TU Datasets
# TU Datasets: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
graph_dicts = tfg.datasets.TUDataset("NCI1").load_data()


# ==================================== Basic OOP API ====================================
# OOP Style GCN (Graph Convolutional Network)
gcn_layer = tfg.layers.GCN(units=20, activation=tf.nn.relu)

for graph in test_data:
    # Cache can speed-up GCN by caching the normed edge information
    outputs = gcn_layer([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)
    print(outputs)


# OOP Style GAT (Multi-head Graph Attention Network)
gat_layer = tfg.layers.GAT(units=20, activation=tf.nn.relu, num_heads=4)
for graph in test_data:
    outputs = gat_layer([graph.x, graph.edge_index])
    print(outputs)


# OOP Style Multi-layer GCN Model
class GCNModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn0 = tfg.layers.GCN(16, activation=tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(7)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, edge_weight = inputs
        h = self.dropout(x, training=training)
        h = self.gcn0([h, edge_index, edge_weight], cache=cache)
        h = self.dropout(h, training=training)
        h = self.gcn1([h, edge_index, edge_weight], cache=cache)
        return h


gcn_model = GCNModel()
for graph in test_data:
    outputs = gcn_model([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)
    print(outputs)


# ==================================== Basic Functional API ====================================
# Functional Style GCN
# Functional API is more flexible for advanced algorithms
# You can pass both data and parameters to functional APIs

gcn_w = tf.Variable(tf.random.truncated_normal([test_data[0].num_features, 20]))
for graph in test_data:
    outputs = tfg.nn.gcn(graph.x, graph.adj(), gcn_w, activation=tf.nn.relu)
    print(outputs)


# ==================================== Advanced Functional API ====================================
# Most APIs are implemented with Map-Reduce Style
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




## Cite

If you use tf_geometric in a scientific publication, we would appreciate citations to the following paper:

```html
@inproceedings{DBLP:conf/mm/HuQFWZZX21,
  author    = {Jun Hu and
               Shengsheng Qian and
               Quan Fang and
               Youze Wang and
               Quan Zhao and
               Huaiwen Zhang and
               Changsheng Xu},
  editor    = {Heng Tao Shen and
               Yueting Zhuang and
               John R. Smith and
               Yang Yang and
               Pablo Cesar and
               Florian Metze and
               Balakrishnan Prabhakaran},
  title     = {Efficient Graph Deep Learning in TensorFlow with tf{\_}geometric},
  booktitle = {{MM} '21: {ACM} Multimedia Conference, Virtual Event, China, October
               20 - 24, 2021},
  pages     = {3775--3778},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3474085.3478322},
  doi       = {10.1145/3474085.3478322},
  timestamp = {Wed, 20 Oct 2021 12:40:01 +0200},
  biburl    = {https://dblp.org/rec/conf/mm/HuQFWZZX21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


## Related Projects

+ __tf_sparse:__ We develop [TensorFlow Sparse (tf_sparse)](https://github.com/CrawlScript/tf_sparse) to implement efficient and elegant 
sparse TensorFlow operations for tf_geometric. URL: [https://github.com/CrawlScript/tf_sparse](https://github.com/CrawlScript/tf_sparse).
+ __GRecX:__ [GRecX](https://github.com/maenzhier/GRecX) is an efficient and unified benchmark for GNN-based recommendation. URL: [https://github.com/maenzhier/GRecX](https://github.com/maenzhier/GRecX).
