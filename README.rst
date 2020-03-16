
tf_geometric
============

Efficient and Friendly Graph Neural Network Library for TensorFlow 1.x and 2.x.

Inspired by **rusty1s/pytorch_geometric**\ , we build a GNN library for TensorFlow.

Homepage and Documentation
--------------------------


* Homepage: `https://github.com/CrawlScript/tf_geometric <https://github.com/CrawlScript/tf_geometric>`_
* Documentation: `https://tf-geometric.readthedocs.io <https://tf-geometric.readthedocs.io>`_

Efficient and Friendly
----------------------

We use Message Passing mechanism to implement graph neural networks, which is way efficient than the dense matrix based implementations and more friendly than the sparse matrix based ones.
In addition, we provide easy and elegant APIs for complex GNN operations.
The following example constructs a graph and applies a Multi-head Graph Attention Network (GAT) on it:

.. code-block:: python

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

Output:

.. code-block::

   Graph Desc:
    Graph Shape: x => (5, 20)  edge_index => (2, 4)    y => None

   Processed Graph Desc:
    Graph Shape: x => (5, 20)  edge_index => (2, 8)    y => None

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

DEMO
----

We recommend you to get started with some demo.

Node Classification
^^^^^^^^^^^^^^^^^^^


* `Graph Convolutional Network (GCN) <demo/demo_gcn.py>`_
* `Multi-head Graph Attention Network (GAT) <demo/demo_gat.py>`_

Link Prediction
^^^^^^^^^^^^^^^


* `MeanPooling <demo/demo_mean_pool.py>`_
* `SAGPooling <demo/demo_sag_pool_h.py>`_

Graph Classification
^^^^^^^^^^^^^^^^^^^^


* `Graph Auto-Encoder (GAE) <demo/demo_gae.py>`_

Installation
------------

Requirements:


* Operation System: Windows / Linux / Mac OS
* Python: version >= 3.5
* Python Packages:

  * tensorflow/tensorflow-gpu: >= 1.14.0 or >= 2.0.0b1
  * numpy >= 1.17.4
  * networkx >= 2.1
  * scipy >= 1.1.0

Use one of the following commands below:

.. code-block:: bash

   pip install -U tf_geometric # this will not install the tensorflow/tensorflow-gpu package

   pip install -U tf_geometric[tf1-cpu] # this will install TensorFlow 1.x CPU version

   pip install -U tf_geometric[tf1-gpu] # this will install TensorFlow 1.x GPU version

   pip install -U tf_geometric[tf2-cpu] # this will install TensorFlow 2.x CPU version

   pip install -U tf_geometric[tf2-gpu] # this will install TensorFlow 2.x GPU version

OOP and Functional API
----------------------

We provide both OOP and Functional API, with which you can make some cool things.

.. code-block:: python

   # coding=utf-8
   import os

   # Enable GPU 0
   os.environ["CUDA_VISIBLE_DEVICES"] = "0"

   import tf_geometric as tfg
   import tensorflow as tf
   import numpy as np
   from tf_geometric.utils.graph_utils import convert_edge_to_directed

   # ==================================== Graph Data Structure ====================================
   # In tf_geometric, graph data can be either individual Tensors or Graph objects
   # A graph usually consists of x(node features), edge_index and edge_weight(optional)

   # Node Features => (num_nodes, num_features)
   x = np.random.randn(5, 20).astype(np.float32) # 5 nodes, 20 features

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

   # Make the edge_index directed such that we can use it as the input of GCN
   edge_index, [edge_weight] = convert_edge_to_directed(edge_index, [edge_weight])


   # We can convert these numpy array as TensorFlow Tensors and pass them to gnn functions
   outputs = tfg.nn.gcn(
       tf.Variable(x),
       tf.constant(edge_index),
       tf.constant(edge_weight),
       tf.Variable(tf.random.truncated_normal([20, 2])) # GCN Weight
   )
   print(outputs)

   # Usually, we use a graph object to manager these information
   # edge_weight is optional, we can set it to None if you don't need it
   graph = tfg.Graph(x=x, edge_index=edge_index, edge_weight=edge_weight)

   # You can easily convert these numpy arrays as Tensors with the Graph Object API
   graph.convert_data_to_tensor()

   # Then, we can use them without too many manual conversion
   outputs = tfg.nn.gcn(
       graph.x,
       graph.edge_index,
       graph.edge_weight,
       tf.Variable(tf.random.truncated_normal([20, 2])),  # GCN Weight
       cache=graph.cache  # GCN use caches to avoid re-computing of the normed edge information
   )
   print(outputs)


   # For algorithms that deal with batches of graphs, we can pack a batch of graph into a BatchGraph object
   # Batch graph wrap a batch of graphs into a single graph, where each nodes has an unique index and a graph index.
   # The node_graph_index is the index of the corresponding graph for each node in the batch.
   # The edge_graph_index is the index of the corresponding edge for each node in the batch.
   batch_graph = tfg.BatchGraph.from_graphs([graph, graph, graph, graph])

   # We can reversely split a BatchGraph object into Graphs objects
   graphs = batch_graph.to_graphs()

   # Graph Pooling algorithms often rely on such batch data structure
   # Most of them accept a BatchGraph's data as input and output a feature vector for each graph in the batch
   outputs = tfg.nn.mean_pool(batch_graph.x, batch_graph.node_graph_index, num_graphs=batch_graph.num_graphs)
   print(outputs)

   # We provide some advanced graph pooling operations such as topk_pool
   node_score = tfg.nn.gcn(
       batch_graph.x,
       batch_graph.edge_index,
       batch_graph.edge_weight,
       tf.Variable(tf.random.truncated_normal([20, 1])),  # GCN Weight
       cache=graph.cache  # GCN use caches to avoid re-computing of the normed edge information
   )
   node_score = tf.reshape(node_score, [-1])
   topk_node_index = tfg.nn.topk_pool(batch_graph.node_graph_index, node_score, ratio=0.6)
   print(topk_node_index)




   # ==================================== Built-in Datasets ====================================
   # all graph data are in numpy format
   train_data, valid_data, test_data = tfg.datasets.PPIDataset().load_data()

   # we can convert them into tensorflow format
   test_data = [graph.convert_data_to_tensor() for graph in test_data]





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



   # ==================================== Basic Functional API ====================================
   # Functional Style GCN
   # Functional API is more flexible for advanced algorithms
   # You can pass both data and parameters to functional APIs

   gcn_w = tf.Variable(tf.random.truncated_normal([test_data[0].num_features, 20]))
   for graph in test_data:
       outputs = tfg.nn.gcn(graph.x, edge_index, edge_weight, gcn_w, activation=tf.nn.relu)
       print(outputs)


   # ==================================== Advanced OOP API ====================================
   # All APIs are implemented with Map-Reduce Style
   # This is a gcn without weight normalization and transformation.
   # Create your own GNN Layer by subclassing the MapReduceGNN class
   class NaiveGCN(tfg.layers.MapReduceGNN):

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
