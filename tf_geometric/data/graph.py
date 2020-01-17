# coding=utf-8
from keras.datasets import mnist
import tensorflow as tf


class Graph(object):
    def __init__(self, x=None, edge_index=None, y=None,
                 edge_weight=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.y = y

        self.cached_normed_edge_weight = None

    @property
    def num_nodes(self):
        return len(self.x)

    @property
    def num_features(self):
        return self.x.shape[-1]

    def get_shape(self, data):
        return None if data is None else data.shape

    def get_shape_desc(self):
        return "Graph Shape: x => {}\tedge_index => {}\ty => {}".format(
            self.get_shape(self.x),
            self.get_shape(self.edge_index),
            self.get_shape(self.y)
        )

    def __str__(self):
        return self.get_shape_desc()

    def convert_data_to_tensor(self):
        for key in ["x", "edge_index", "edge_weight", "y"]:
            data = getattr(self, key)

            if data is not None and not tf.is_tensor(data):
                setattr(self, key, tf.convert_to_tensor(data))
        return self
