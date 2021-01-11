# coding=utf-8
import tensorflow as tf
from tf_geometric.nn import aggregate_neighbors


class MapReduceGNN(tf.keras.Model):

    def map(self, repeated_x, neighbor_x, edge_weight=None):
        pass

    def reduce(self, neighbor_msg, node_index, num_nodes=None):
        pass

    def update(self, x, reduced_neighbor_msg):
        pass

    def get_mapper(self):
        def mapper(repeated_x, neighbor_x, edge_weight=None):
            return self.map(repeated_x, neighbor_x, edge_weight)
        return mapper

    def get_reducer(self):
        def reducer(neighbor_msg, node_index, num_nodes=None):
            return self.reduce(neighbor_msg, node_index, num_nodes)
        return reducer

    def get_updater(self):
        def updater(x, reduced_neighbor_msg):
            return self.update(x, reduced_neighbor_msg)
        return updater

    def call(self, inputs, training=None, mask=None):
        x, edge_index, edge_weight = inputs
        return aggregate_neighbors(
            x,
            edge_index,
            edge_weight,
            self.get_mapper(),
            self.get_reducer(),
            self.get_updater()
        )