Quickstart by Examples
======================


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


* `Graph Convolutional Network (GCN) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gcn.py>`_
* `Multi-head Graph Attention Network (GAT) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gat.py>`_

Link Prediction
^^^^^^^^^^^^^^^


* `MeanPooling <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_mean_pool.py>`_
* `SAGPooling <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_sag_pool_h.py>`_

Graph Classification
^^^^^^^^^^^^^^^^^^^^


* `Graph Auto-Encoder (GAE) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gae.py>`_