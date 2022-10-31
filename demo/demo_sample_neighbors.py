# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_geometric.utils.graph_utils import reindex_sampled_edge_index, RandomNeighborSampler
from tf_geometric.datasets import CoraDataset
import time
import numpy as np
import tf_geometric as tfg

edge_index = [
    [0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 5, 5, 5],
    [1, 2, 3, 4, 5, 0, 4, 1, 2, 0, 2, 3, 4]
]

neighbor_sampler = RandomNeighborSampler(edge_index)
sampled_edge_index, sampled_edge_weight  = neighbor_sampler.sample(k=5, central_node_index=None, padding=False)
print(sampled_edge_index, sampled_edge_weight)


graph, (train_index, valid_index, test_index) = CoraDataset().load_data()
neighbor_sampler = RandomNeighborSampler(graph.edge_index)

for _ in range(100):
    start = time.time()
    sampled_edge_index, sampled_edge_weight = neighbor_sampler.sample(ratio=0.5)
    print(sampled_edge_index, sampled_edge_weight)
    print(time.time() - start)


for _ in range(100):
    start = time.time()
    print("sample for central nodes: ")
    central_node_index = np.arange(100, 200)
    sampled_edge_index, sampled_edge_weight = neighbor_sampler.sample(ratio=0.5, central_node_index=central_node_index)
    print("central_node_index: ", central_node_index)
    print("sampled_edge_index: \n", sampled_edge_index)
    print("sampled_edge_weight: ", sampled_edge_weight)

    # print("reindex sampled nodes and edges to construct edges for a subgraph: ")
    # reindexed_edge_index = reindex_sampled_edge_index(sampled_edge_index, sampled_node_index)
    # print("reindexed_edge_index: \n", reindexed_edge_index)
    # print(time.time() - start, "\n")
