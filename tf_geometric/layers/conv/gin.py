# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.conv.gin import gin


class GIN(tf.keras.Model):
    """
    Graph Isomorphism Network  Layer
    """

    def __init__(self, mlp_model, eps=0, train_eps=False, *args, **kwargs):
        """
        :param mlp_model: A neural network (multi-layer perceptrons).
        :param eps: float, optional, (default: :obj:`0.`).
        :param train_eps: Boolean, Whether the eps is trained.
        :param activation: Activation function to use.
        """
        super().__init__(*args, **kwargs)
        self.mlp_model = mlp_model

        self.eps = eps
        if train_eps:
            self.eps = self.add_weight("eps", shape=[], initializer="zeros")


    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GIN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, _ = inputs
        else:
            x, edge_index = inputs

        return gin(x, edge_index, self.mlp_model, self.eps, training=training)
