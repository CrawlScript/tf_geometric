.. _index:

tf_geometric Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

----

:ref:`(中文版)<index_cn>`


Efficient and Friendly Graph Neural Network Library for TensorFlow 1.x and 2.x.


Inspired by **rusty1s/pytorch_geometric**\ , we build a GNN library for TensorFlow.
`tf_geometric <https://github.com/CrawlScript/tf_geometric>`_ provides both OOP and Functional API, with which you can make some cool things.


* **Github:**\  `https://github.com/CrawlScript/tf_geometric <https://github.com/CrawlScript/tf_geometric>`_
* **Documentation:**\  `https://tf-geometric.readthedocs.io <https://tf-geometric.readthedocs.io>`_
* **Paper:**\  `Efficient Graph Deep Learning in TensorFlow with tf_geometric <https://arxiv.org/abs/2101.11552>`_


.. raw:: html

   <p align="center">
   <img src="https://raw.githubusercontent.com/CrawlScript/tf_geometric/master/TF_GEOMETRIC_LOGO.png" style="max-width: 400px; width: 100%;"/>
   </p>






Efficient and Friendly API
----------------------------------------------------------

----

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

.. code-block:: HTML

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


Tutorial
----------

----


Table of Contents
>>>>>>>>>>>>>>>>>>>


.. toctree::
   :maxdepth: 2

   wiki/installation
   wiki/quickstart



Getting Started with Demo
>>>>>>>>>>>>>>>>>>>>>>>>>


We recommend you to get started with some demo.

Node Classification
:::::::::::::::::::

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
:::::::::::::::::::

* `MeanPooling <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_mean_pool.py>`_
* `Graph Isomorphism Network (GIN) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gin.py>`_
* `Self-Attention Graph Pooling (SAGPooling) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_sag_pool_h.py>`_
* `Hierarchical Graph Representation Learning with Differentiable Pooling (DiffPool) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_diff_pool.py>`_
* `Order Matters: Sequence to Sequence for Sets (Set2Set) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_set2set.py>`_
* `ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations (ASAP) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_asap.py>`_
* `An End-to-End Deep Learning Architecture for Graph Classification (SortPool) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_sort_pool.py>`_
* `Spectral Clustering with Graph Neural Networks for Graph Pooling (MinCutPool) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_min_cut_pool.py>`_



Link Prediction
:::::::::::::::::::

* `Graph Auto-Encoder (GAE) <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gae.py>`_



Save and Load Models
::::::::::::::::::::

* `Save and Load Models <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_save_and_load_model.py>`_
* `Save and Load Models with tf.train.Checkpoint <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_checkpoint.py>`_



Distributed Training
::::::::::::::::::::

* `Distributed GCN for Node Classification <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_distributed_gcn.py>`_
* `Distributed MeanPooling for Graph Classification <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_distributed_mean_pool.py>`_


Sparse
::::::::::::::::::::

* `Sparse Node Features <https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_sparse_node_features.py>`_



Cite
----

If you use tf_geometric in a scientific publication, we would appreciate citations to the following paper:

.. code-block:: html

   @misc{hu2021efficient,
         title={Efficient Graph Deep Learning in TensorFlow with tf_geometric},
         author={Jun Hu and Shengsheng Qian and Quan Fang and Youze Wang and Quan Zhao and Huaiwen Zhang and Changsheng Xu},
         year={2021},
         eprint={2101.11552},
         archivePrefix={arXiv},
         primaryClass={cs.LG}
   }




Package Reference
-----------------

----


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
