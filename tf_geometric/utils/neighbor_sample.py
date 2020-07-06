#encoding='utf-8'

import random
import tensorflow as tf
import numpy as np
from tensorflow import SparseTensor
from typing import List, Optional, Tuple, NamedTuple

class Adj(NamedTuple):
    edge_index: tf.Tensor
    e_id: tf.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)

def get_neighbors(edge_index):

    neighbors = dict()
    to_neighbors = []
    row, col = edge_index
    for i in range(len(row)):
        neighbors.setdefault(row[i], []).append(col[i])

    # neighbors_index = []
    for _, v in neighbors.items():

        to_neighbors.append(v)

    return to_neighbors

def sample_neighbors(to_neighbors, edge_index, edge_weight, num_sample=None):

    row, _ = edge_index
    dic = dict()
    nodes = list(row)
    nodes = list(dic.fromkeys(nodes).keys())
    nodes_index = np.argsort(nodes)

    # sorted_edge_weight = edge_weight[nodes_index]


    _set = set
    _sample = random.sample
    sampled_neighbors = [_sample(to_neigh,
                        num_sample,
                        ) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighbors]

    edge_weight = list(edge_weight)
    removed_nodes_num = 0
    offset = 0
    for i in range(len(to_neighbors)):
        removed_node = list(_set(to_neighbors[i]).difference(_set(sampled_neighbors[i])))
        for node in removed_node:
            _index = to_neighbors[i].index(node)
            del edge_weight[removed_nodes_num + _index - offset]
            offset += 1
        removed_nodes_num += len(to_neighbors[i])



    num_neighbors = 0
    for neighs in sampled_neighbors:
        num_neighbors += len(neighs)
    sampled_edge_index = np.zeros((2, num_neighbors), dtype=np.int)

    pre_node_neighbors_num = 0
    for i in range(len(nodes)):
        n = len(sampled_neighbors[nodes_index[i]])
        sampled_edge_index[0][pre_node_neighbors_num:pre_node_neighbors_num + n] = nodes[nodes_index[i]]
        sampled_edge_index[1][pre_node_neighbors_num:pre_node_neighbors_num + n] = sampled_neighbors[nodes_index[i]]

        pre_node_neighbors_num += n


    return sampled_edge_index, edge_weight






