# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.pool.common_pool import mean_pool, min_pool, max_pool, sum_pool


class CommonPool(tf.keras.Model):

    def __init__(self, pool_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_func = pool_func

    def call(self, inputs, training=None, mask=None):
        if len(inputs) == 2:
            x, node_graph_index = inputs
            num_graphs = None
        else:
            x, node_graph_index, num_graphs = inputs

        return self.pool_func(x, node_graph_index, num_graphs)


class MeanPool(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(mean_pool, *args, **kwargs)


class MinPool(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(min_pool, *args, **kwargs)


class MaxPool(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(max_pool, *args, **kwargs)


class SumPool(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(sum_pool, *args, **kwargs)