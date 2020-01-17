# coding=utf-8
from keras.datasets import mnist


class Graph(object):
    def __init__(self, x=None, edge_index=None, y=None, directed=False):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.directed = directed

    def get_shape(self, data):
        return None if data is None else data.shape

    def print_shape(self):
        return "Graph Shape: x => {}\tedge_index => {}\ty => {}".format(
            self.get_shape(self.x),
            self.get_shape(self.edge_index),
            self.get_shape(self.y)
        )

