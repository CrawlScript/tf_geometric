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


class PlanetoidDataset(DownloadableDataset):

    def __init__(self, dataset_name, task="semi_supervised", dataset_root_path=None):
        """

        :param dataset_name: "cora" | "citeseer" | "pubmed"
        :param task: "semi_supervised" | "supervised"
        :param dataset_root_path:
        """
        valid_tasks = ["semi_supervised", "supervised"]
        if task not in valid_tasks:
            raise Exception("invalid task name for planetoid dataset: \"{}\"\tvalid task names: {}".format(task, valid_tasks))
        self.task = task

        super().__init__(dataset_name=dataset_name,
                         download_urls=[
                             "http://cdn.zhuanzhi.ai/github/{}.zip".format(dataset_name),
                             "https://github.com/CrawlScript/gnn_datasets/raw/master/planetoid/{}.zip".format(
                                 dataset_name)
                         ],
                         download_file_name="{}.zip".format(dataset_name),
                         cache_name=None,
                         dataset_root_path=dataset_root_path,
                         )

    # https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    def process(self):

        dataset_str = self.dataset_name
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

        with open(os.path.join(self.raw_root_path, "ind.{}.test.index".format(dataset_str)), "r",
                  encoding="utf-8") as f:
            test_idx_reorder = [int(line.strip()) for line in f]
            test_idx_range = np.sort(test_idx_reorder)

        if self.dataset_name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = list(range(min(test_idx_reorder), max(test_idx_reorder) + 1))
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        test_index = test_idx_range.tolist()
        if self.task == "semi_supervised":
            train_index = list(range(len(y)))
            valid_index = list(range(len(y), len(y) + 500))
        else:
            train_index = range(len(ally) - 500)
            valid_index = range(len(ally) - 500, len(ally))

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


class CoraDataset(PlanetoidDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__("cora", dataset_root_path=dataset_root_path)


class CiteseerDataset(PlanetoidDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__("citeseer", dataset_root_path=dataset_root_path)


class PubmedDataset(PlanetoidDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__("pubmed", dataset_root_path=dataset_root_path)


class SupervisedCoraDataset(PlanetoidDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__("cora", task="supervised", dataset_root_path=dataset_root_path)


class SupervisedCiteseerDataset(PlanetoidDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__("citeseer", task="supervised", dataset_root_path=dataset_root_path)


class SupervisedPubmedDataset(PlanetoidDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__("pubmed", task="supervised", dataset_root_path=dataset_root_path)
