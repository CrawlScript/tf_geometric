# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_geometric.utils.graph_utils import reindex_sampled_edge_index, RandomNeighborSampler
from tf_geometric.datasets import CoraDataset
import time
import numpy as np
import tf_geometric as tfg
import tf_sparse as tfs

edge_index = [
    [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5],
    [1, 2, 3, 4, 5, 0, 4, 7, 1, 7, 2, 3, 6, 9, 0, 2, 3, 4, 7, 8, 10]
]

neighbor_sampler = RandomNeighborSampler(edge_index)
sampled_virtual_edge_index, sampled_virtual_edge_weight = neighbor_sampler.sample(k=5, sampled_node_index=([2, 4], [2, 6, 7, 8, 9, 10]), padding=False)
print(sampled_virtual_edge_index)
print(sampled_virtual_edge_weight)


graph, (train_index, valid_index, test_index) = CoraDataset().load_data()
neighbor_sampler = RandomNeighborSampler(graph.edge_index)

for _ in range(100):
    start = time.time()
    sampled_virtual_edge_index, sampled_virtual_edge_weight = neighbor_sampler.sample(ratio=0.5)
    print(sampled_virtual_edge_index)
    print(sampled_virtual_edge_weight)
    print(time.time() - start)


for _ in range(100):
    start = time.time()
    print("sample for sampled nodes: ")
    sampled_node_index = np.arange(100, 200)
    sampled_virtual_edge_index, sampled_virtual_edge_weight = neighbor_sampler.sample(ratio=0.5, sampled_node_index=sampled_node_index)
    print("sampled_node_index: ", sampled_node_index)
    print("sampled_virtual_edge_index: \n", sampled_virtual_edge_index)
    print("sampled_virtual_edge_weight: \n", sampled_virtual_edge_weight)

    # print("reindex sampled nodes and edges to construct edges for a subgraph: ")
    # reindexed_edge_index = reindex_sampled_edge_index(sampled_edge_index, sampled_node_index)
    # print("reindexed_edge_index: \n", reindexed_edge_index)
    # print(time.time() - start, "\n")
