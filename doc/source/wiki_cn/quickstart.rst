.. _wiki_cn-quickstart:

快速入门
======================

:ref:`(English Version)<wiki-quickstart>`


使用简单示例快速入门
---------------------------------

tf_geometric使用消息传递机制来实现图神经网络：相比于基于稠密矩阵的实现，它具有更高的效率；相比于基于稀疏矩阵的实现，它具有更友好的API。
除此之外，tf_geometric还为复杂的图神经网络操作提供了简易优雅的API。
下面的示例展现了使用tf_geometric构建一个图结构的数据，并使用多头图注意力网络（Multi-head GAT）对图数据进行处理的流程：

.. code-block:: python

   # coding=utf-8
   import numpy as np
   import tf_geometric as tfg
   import tensorflow as tf

   graph = tfg.Graph(
       x=np.random.randn(5, 20),  # 5个节点, 20维特征
       edge_index=[[0, 0, 1, 3],
                   [1, 2, 2, 1]]  # 4条无向边
   )

   print("Graph Desc: \n", graph)

   graph.convert_edge_to_directed()  # 预处理边数据，将无向边表示转换为有向边表示
   print("Processed Graph Desc: \n", graph)
   print("Processed Edge Index:\n", graph.edge_index)

   # 多头图注意力网络（Multi-head GAT）
   gat_layer = tfg.layers.GAT(units=4, num_heads=4, activation=tf.nn.relu)
   output = gat_layer([graph.x, graph.edge_index])
   print("Output of GAT: \n", output)


输出:

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




面向对象接口（OOP API）和函数式接口（Functional API）
--------------------------------------

`tf_geometric <https://github.com/CrawlScript/tf_geometric>`_ 同时提供面向对象接口（OOP API）和函数式接口（Functional API），你可以用它们来构建有趣的模型。

.. code-block:: python

   # coding=utf-8
   import os

   # 使用 GPU 0
   os.environ["CUDA_VISIBLE_DEVICES"] = "0"

   import tf_geometric as tfg
   import tensorflow as tf
   import numpy as np
   from tf_geometric.utils.graph_utils import convert_edge_to_directed

   # ==================================== 图数据结构 ====================================
   # 在tf_geometric中, 1个图的数据可以被存储为多个张量(numpy.ndarray或tf.Tensor)或一个tfg.Graph对象。
   # 一个图通常包含节点特征x、边表edge_index和边权重edge_weight（可选）。

   # 节点特征 => (num_nodes, num_features)
   x = np.random.randn(5, 20).astype(np.float32)  # 5个节点，20维特征

   # 边表 => (2, num_edges)
   # 边表edge_index中的每列(u, v)表示一个从节点u到v的有向边。
   # 注意，它并不包含从v到u的边，你需要在边表中提供(v, u)来表示从v到u的边。
   # 有时候这种表示方法并不方便，因为对于每条无向边，用户需要同时提供两列数据。
   # 为此，我们允许用户仅提供单向的边（无向图表示形式），并在之后使用工具方法将其转换为同时包含双向边的edge_index（有向图表示法）
   # 也就是说，用户仅在edge_index中提供(u, v)，然后使用`convert_edge_to_directed`方法将其转换为(u, v)和(v, u)。
   edge_index = np.array([
       [0, 0, 1, 3],
       [1, 2, 2, 1]
   ])

   # 边权重 => (num_edges)
   edge_weight = np.array([0.9, 0.8, 0.1, 0.2]).astype(np.float32)

   # 将edge_index从无向图表示法转换为有向图表示法，这样才可以将其作为图卷积网络GCN的输入
   edge_index, [edge_weight] = convert_edge_to_directed(edge_index, [edge_weight])


   # 可以将numpy张量转换为TensorFlow张量，并将其作为函数式API（Functional API）的输入
   outputs = tfg.nn.gcn(
       tf.Variable(x),
       tf.constant(edge_index),
       tf.constant(edge_weight),
       tf.Variable(tf.random.truncated_normal([20, 2])) # GCN Weight
   )
   print(outputs)

   # 通常，可以用一个tfg.Graph对象来维护一个图的信息
   # 其中，边权重edge_weight是可选的，可以将其设置为None
   graph = tfg.Graph(x=x, edge_index=edge_index, edge_weight=edge_weight)

   # 如果有必要，可以用tfg.Graph对象的`convert_data_to_tensor`方法直接将图中的numpy数据转换为TensorFlow张量
   graph.convert_data_to_tensor()

   # 转换之后，我们可以直接将图的属性作为函数式API（Functional API）的输入
   outputs = tfg.nn.gcn(
       graph.x,
       graph.edge_index,
       graph.edge_weight,
       tf.Variable(tf.random.truncated_normal([20, 2])),  # GCN Weight
       cache=graph.cache  # 图卷积网络层GCN使用缓存cache来避免对归一化边信息的重复计算
   )
   print(outputs)


   # 对于需要批量处理图的算法，可以将批量的图（多图）打包进一个tfg.BatchGraph对象。
   # tfg.BatchGraph将一批图打包为一个单独的大图，原始批量图中的每个节点在大图中都有独立的索引号以及图索引号（表示属于第几个原始图）
   # tfg.BatchGraph对象的node_graph_index属性表示大图中每个节点所对应的原始图索引号。
   # tfg.BatchGraph对象的edge_graph_index属性表示大图中每条边所对应的原始图索引号。
   batch_graph = tfg.BatchGraph.from_graphs([graph, graph, graph, graph])

   # 也可以逆向地将tfg.BatchGraph对象拆分为多个tfg.Graph对象
   graphs = batch_graph.to_graphs()

   # 图池化操作通常会依赖于tfg.BatchGraph
   # 大多图池化操作以1个tfg.BatchGraph对象的属性作为输入，为批量图中的每个图输出1个特征向量作为每个图的表示
   outputs = tfg.nn.mean_pool(batch_graph.x, batch_graph.node_graph_index, num_graphs=batch_graph.num_graphs)
   print(outputs)

   # 框架也提供了一些高阶的图池化操作，例如topk_pool
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




   # ==================================== 内置数据集 ====================================
   # 内置数据集通常为numpy格式
   train_data, valid_data, test_data = tfg.datasets.PPIDataset().load_data()

   # 如果需要，可以将其转换为TensorFlow张量
   test_data = [graph.convert_data_to_tensor() for graph in test_data]





   # ======================== 基础的面向对象API（Basic OOP API）======================== 
   # 面向对象风格的图卷积网络层GCN
   gcn_layer = tfg.layers.GCN(units=20, activation=tf.nn.relu)

   for graph in test_data:
       # 使用缓存cache可以避免对归一化边信息的重复计算，大幅度加速GCN的计算
       outputs = gcn_layer([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)
       print(outputs)


   # OOP Style GAT (Multi-head Graph Attention Network)
   # 面向对象风格的多头图注意力网络GAT
   gat_layer = tfg.layers.GAT(units=20, activation=tf.nn.relu, num_heads=4)
   for graph in test_data:
       outputs = gat_layer([graph.x, graph.edge_index])
       print(outputs)


   # 面向对象风格的多层图卷积网络模型（Multi-layer GCN Model）
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


   # ==================================== 基础的函数式API（Functional API） ====================================
   # 函数式风格的图卷积网络GCN
   # 函数式API对于一些高阶算法会显得更加灵活
   # 你可以同时将数据和网络参数作为函数式API的输入

   gcn_w = tf.Variable(tf.random.truncated_normal([test_data[0].num_features, 20]))
   for graph in test_data:
       outputs = tfg.nn.gcn(graph.x, edge_index, edge_weight, gcn_w, activation=tf.nn.relu)
       print(outputs)



   # ==================================== 进阶的函数式API（Functional API） ====================================
   # 大部分API都是按照Map-Reduce风格实现的
   # 下面实现了一个不包含边归一化和特征变换的图卷积层
   # 只需要将mapper/reducer/updater函数分别传给函数式API中的tfg.nn.aggregate_neighbors方法，即可轻松实现GNN层

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


