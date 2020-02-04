Installation
============


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