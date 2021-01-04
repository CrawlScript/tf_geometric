.. tf_geometric documentation master file, created by
   sphinx-quickstart on Tue Feb  4 15:59:33 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tf_geometric Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Efficient and Friendly Graph Neural Network Library for TensorFlow 1.x and 2.x.


Inspired by **rusty1s/pytorch_geometric**\ , we build a GNN library for TensorFlow.
`tf_geometric <https://github.com/CrawlScript/tf_geometric>`_ provide both OOP and Functional API, with which you can make some cool things.

* **Github:**\  `https://github.com/CrawlScript/tf_geometric <https://github.com/CrawlScript/tf_geometric>`_
* **Documentation:**\  `https://tf-geometric.readthedocs.io <https://tf-geometric.readthedocs.io>`_



Efficient and Friendly Graph Data Structure and GNN Layers
----------------------------------------------------------

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


Quickstart
----------

.. toctree::
   :maxdepth: 2

   wiki/installation
   wiki/quickstart
   wiki/demo



Getting Started with Demo
-------------------------

We recommend you to get started with some demo.

Node Classification
>>>>>>>>>>>>>>>>>>>

* `Graph Convolutional Network (GCN) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gcn.py>`_
* `Multi-head Graph Attention Network (GAT) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gat.py>`_
* `Approximate Personalized Propagation of Neural Predictions (APPNP) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_appnp.py>`_
* `Inductive Representation Learning on Large Graphs (GraphSAGE) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_graph_sage.py>`_
* `Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (ChebyNet) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_chebynet.py>`_
* `Simple Graph Convolution (SGC) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_sgc.py>`_
* `Topology Adaptive Graph Convolutional Network (TAGCN) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_tagcn.py>`_
* `Deep Graph Infomax (DGI) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_dgi.py>`_
* `DropEdge: Towards Deep Graph Convolutional Networks on Node Classification (DropEdge) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_drop_edge_gcn.py>`_
* `Graph Convolutional Networks for Text Classification (TextGCN) <https://github.com/CrawlScript/TensorFlow-TextGCN>`_


Graph Classification
>>>>>>>>>>>>>>>>>>>>

* `MeanPooling <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_mean_pool.py>`_
* `Graph Isomorphism Network (GIN) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gin.py>`_
* `SAGPooling <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_sag_pool_h.py>`_



Link Prediction
>>>>>>>>>>>>>>>

* `Graph Auto-Encoder (GAE) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gae.py>`_



Save and Load Models
>>>>>>>>>>>>>>>>>>>>

* `Save and Load Models <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_save_and_load_model.py>`_
* `Save and Load Models with tf.train.Checkpoint <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_checkpoint.py>`_



Package Reference
-----------------


.. _Packages:

.. toctree::
   :glob:
   :maxdepth: 2

   modules/root
   modules/datasets
   modules/layers
   modules/nn
   modules/utils




..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
