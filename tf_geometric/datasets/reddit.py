# coding=utf-8

import numpy as np
import os

import networkx as nx
from tf_geometric.data.graph import Graph
from tf_geometric.data.dataset import DownloadableDataset
import json
import scipy.sparse as sp
from tf_geometric.utils.data_utils import load_cache
from tf_geometric.utils.graph_utils import convert_edge_to_directed


class _BaseRedditDataset(DownloadableDataset):

    def __init__(self, dataset_root_path=None, cache_name=None):
        super().__init__(dataset_name="reddit",
                         download_urls=[
                             "https://data.dgl.ai/dataset/reddit.zip"
                         ],
                         download_file_name="reddit.zip",
                         cache_name=cache_name,
                         dataset_root_path=dataset_root_path,
                         )

    def process(self):

        common_data_path = os.path.join(self.raw_root_path, "reddit_data.npz")
        common_data = np.load(common_data_path)

        x = common_data["feature"]
        y = common_data["label"]

        mask = common_data["node_types"]
        full_index = np.arange(len(x), dtype=np.int32)
        train_index = full_index[mask == 1]
        valid_index = full_index[mask == 2]
        test_index = full_index[mask == 3]

        graph_data_path = os.path.join(self.raw_root_path, "reddit_graph.npz")

        adj = sp.load_npz(graph_data_path)
        edge_index = np.stack([adj.row, adj.col], axis=0)

        graph = Graph(x=x, y=y, edge_index=edge_index)
        return graph, (train_index, valid_index, test_index)


class TransductiveRedditDataset(_BaseRedditDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_root_path=dataset_root_path, cache_name="transductive_cache.p")


class InductiveRedditDataset(_BaseRedditDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_root_path=dataset_root_path, cache_name="inductive_cache.p")

    def process(self):
        graph, (train_index, valid_index, test_index) = super().process()
        train_graph = graph.sample_new_graph_by_node_index(train_index)
        valid_graph = graph.sample_new_graph_by_node_index(valid_index)
        test_graph = graph.sample_new_graph_by_node_index(test_index)
        return train_graph, valid_graph, test_graph












