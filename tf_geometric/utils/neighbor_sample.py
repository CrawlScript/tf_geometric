#encoding='utf-8'

import random
import tensorflow as tf
import numpy as np
from collections import defaultdict
from typing import List, Optional, Tuple, NamedTuple


def get_neighbors(edge_index):

    neighbors = defaultdict(list)
    to_neighbors = []
    row, col = edge_index
    for i in range(len(row)):
        # neighbors.setdefault(row[i], []).append(col[i])
        neighbors[row[i]].append(col[i])

    # neighbors_index = []
    for _, v in neighbors.items():

        to_neighbors.append(v)

    return to_neighbors

def sample_neighbors(to_neighbors, edge_index, edge_weight, num_sample=None):

    row, _ = edge_index
    nodes = list(set(row))

    _set = set
    _sample = random.sample
    sampled_neighbors = [_sample(to_neigh,
                        num_sample,
                        ) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighbors]



    num_neighbors = 0
    for neighs in sampled_neighbors:
        num_neighbors += len(neighs)
    sampled_edge_index = np.zeros((2, num_neighbors), dtype=np.int)

    pre_node_neighbors_num = 0
    for i in range(len(sampled_neighbors)):
        n = len(sampled_neighbors[i])
        sampled_edge_index[0][pre_node_neighbors_num:pre_node_neighbors_num + n] = nodes[i]
        sampled_edge_index[1][pre_node_neighbors_num:pre_node_neighbors_num + n] = sampled_neighbors[i]

        pre_node_neighbors_num += n

    edge_weight = np.ones([sampled_edge_index.shape[1]], dtype=np.float32)

    return sampled_edge_index, edge_weight



def sorted_edge_index(graph, to_neighbors):
    row, _ = graph.edge_index
    dic = dict()
    nodes = list(row)
    nodes = list(dic.fromkeys(nodes).keys())
    nodes_index = np.argsort(nodes)

    sorted_edge_index = np.zeros((2, len(row)), dtype=np.int)

    pre_node_neighbors_num = 0
    for i in range(len(nodes)):
        n = len(to_neighbors[nodes_index[i]])
        sorted_edge_index[0][pre_node_neighbors_num:pre_node_neighbors_num + n] = nodes[nodes_index[i]]
        sorted_edge_index[1][pre_node_neighbors_num:pre_node_neighbors_num + n] = to_neighbors[nodes_index[i]]

        pre_node_neighbors_num += n

    edge_weight = np.ones([sorted_edge_index.shape[1]], dtype=np.float32)

    return sorted_edge_index, edge_weight



