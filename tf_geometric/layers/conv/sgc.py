# coding=utf-8

from tf_geometric.nn.conv.sgc import sgc
from tf_geometric.layers.kernel.map_reduce import MapReduceGNN

class SGC(MapReduceGNN):
    """
    The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper
    """

    def __init__(self, units, k=1, use_bias = True, renorm=True, improved=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        :param units: Size of each output sample..
        :param k: Number of hops.(default: :obj:`1`)
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
        :param improved: Whether use improved GCN or not.
        """

        self.units = units
        self.use_bias = use_bias
        self.renorm = renorm
        self.improved = improved
        self.K = k
        self.kernel = []
        self.bias = []

    def build(self, input_shape):

        x_shape = input_shape[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units], initializer="glorot_uniform")
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

    def call(self, inputs, cache=None, training=None, mask=None):
        """
        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, num_units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return sgc(x, edge_index, edge_weight, self.K, self.kernel, self.bias, self.renorm, self.improved, cache)


