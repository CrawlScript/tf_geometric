.. _wiki_cn-installation:

安装
============

:ref:`(English Version)<wiki-installation>`


环境要求与依赖库
------------

* Operation System: Windows / Linux / Mac OS
* Python: version >= 3.5 and version != 3.6
* Python Packages:

  * tensorflow/tensorflow-gpu: >= 1.15.0 or >= 2.3.0
  * numpy >= 1.17.4
  * networkx >= 2.1
  * scipy >= 1.1.0



使用pip一键安装tf_geometric及依赖
------------

使用下面任意一条pip命令进行安装：


.. code-block:: bash

   pip install -U tf_geometric # 不会额外安装tensorflow或tensorflow-gpu包

   pip install -U tf_geometric[tf1-cpu] # 会额外安装TensorFlow 1.x CPU版 

   pip install -U tf_geometric[tf1-gpu] # 会额外安装TensorFlow 1.x GPU版

   pip install -U tf_geometric[tf2-cpu] # 会额外安装TensorFlow 2.x CPU版

   pip install -U tf_geometric[tf2-gpu] # 会额外安装TensorFlow 2.x GPU版
