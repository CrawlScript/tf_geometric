# coding=utf-8
import os

from tf_geometric.utils.graph_utils import RandomNeighborSampler, reindex_sampled_edge_index, RandomNeighborSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tf_geometric.datasets import CoraDataset
import time

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
    central_node_index = [0, 1, 5, 24]
    (sampled_edge_index, sampled_edge_weight), (sampled_node_index, central_node_index, non_central_node_index) = \
        neighbor_sampler.sample(ratio=0.5, central_node_index=central_node_index)
    print("sampled_node_index: ", sampled_node_index)
    print("central_node_index: ", central_node_index)
    print("non_central_node_index: ", non_central_node_index)
    print("sampled_edge_index: \n", sampled_edge_index)
    print("sampled_edge_weight: ", sampled_edge_weight)

    print("reindex sampled nodes and edges to construct edges for a subgraph: ")
    reindexed_edge_index = reindex_sampled_edge_index(sampled_edge_index, sampled_node_index)
    print("reindexed_edge_index: \n", reindexed_edge_index)
    print(time.time() - start, "\n")
