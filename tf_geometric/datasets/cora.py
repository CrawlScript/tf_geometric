# coding=utf-8

import numpy as np
from tf_geometric.data.graph import Graph
from tf_geometric.data.dataset import DownloadableDataset
import scipy.sparse as sp
import os
import sys
import pickle
import networkx as nx

from tf_geometric.utils.graph_utils import convert_edge_to_directed, remove_self_loop_edge


class CoraDataset(DownloadableDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="Cora",
                         download_urls=[
                             "http://cdn.zhuanzhi.ai/github/cora.zip",
                             "https://github.com/CrawlScript/gnn_datasets/raw/master/CORA/cora.zip"
                         ],
                         download_file_name="cora.zip",
                         cache_name=None,
                         dataset_root_path=dataset_root_path,
                         )


    def process(self):

        dataset_str = "cora"
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            data_name = "ind.{}.{}".format(dataset_str, names[i])
            data_path = os.path.join(self.raw_root_path, data_name)
            with open(data_path, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pickle.load(f, encoding='latin1'))
                else:
                    objects.append(pickle.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        with open(os.path.join(self.raw_root_path, "ind.{}.test.index".format(dataset_str)), "r", encoding="utf-8") as f:
            test_idx_reorder = [int(line.strip()) for line in f]
            test_idx_range = np.sort(test_idx_reorder)


        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        test_index = test_idx_range.tolist()
        train_index = list(range(len(y)))
        valid_index = list(range(len(y), len(y)+500))

        x = np.array(features.todense()).astype(np.float32)
        inv_sum_x = 1.0 / np.sum(x, axis=-1, keepdims=True)
        inv_sum_x[np.isnan(inv_sum_x)] = 1.0
        inv_sum_x[np.isinf(inv_sum_x)] = 1.0
        x *= inv_sum_x

        edge_index = np.array(nx.from_dict_of_lists(graph).edges).T
        edge_index, _ = remove_self_loop_edge(edge_index)
        edge_index, _ = convert_edge_to_directed(edge_index)
        y = np.argmax(labels, axis=-1).astype(np.int32)

        graph = Graph(x=x, edge_index=edge_index, y=y)

        return graph, (train_index, valid_index, test_index)