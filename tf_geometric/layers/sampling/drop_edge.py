# coding=utf-8

import tensorflow as tf
from tf_geometric.nn.sampling.drop_edge import drop_edge


class DropEdge(tf.keras.Model):
    def __init__(self, rate=0.5, force_undirected: bool = False):
        """
        DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
        https://openreview.net/forum?id=Hkx1qkrKPr

        :param rate: dropout rate
        :param force_undirected: If set to `True`, will either
            drop or keep both edges of an undirected edge.
        """
        super().__init__()
        self.rate = rate
        self.force_undirected = force_undirected

        if self.rate < 0. or self.rate > 1.:
            raise ValueError('Dropout probability has to be between 0 and 1, '
                             'but got {}'.format(self.rate))

    def call(self, inputs, training=None, mask=None):
        return drop_edge(inputs=inputs, rate=self.rate,
                         force_undirected=self.force_undirected, training=training)
