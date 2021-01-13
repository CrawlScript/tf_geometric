# coding=utf-8
import os

from tf_geometric.utils.graph_utils import RandomNeighborSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tf_geometric.datasets import CoraDataset
import time

graph, (train_index, valid_index, test_index) = CoraDataset().load_data()

neighbor_sampler = RandomNeighborSampler(graph.edge_index)


for _ in range(100):
    start = time.time()
    print(neighbor_sampler.sample(ratio=0.5))
    print(time.time() - start)