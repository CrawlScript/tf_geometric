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


class AmazonElectronicsDataset(DownloadableDataset):

    def __init__(self, dataset_name, dataset_root_path=None):
        """

        :param dataset_name: "amazon-computers" | "amazon-photo"
        :param task: "semi_supervised" | "supervised"
        :param dataset_root_path:
        """
        super().__init__(dataset_name=dataset_name,
                         download_urls=[
                             "https://github.com/CrawlScript/gnn_datasets/raw/master/AmazonElectronics/{}.zip".format(dataset_name),
                             "http://cdn.zhuanzhi.ai/github/{}.zip".format(dataset_name)
                         ],
                         download_file_name="{}.zip".format(dataset_name),
                         cache_name=None,
                         dataset_root_path=dataset_root_path,
                         )

    # https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    def process(self):

        npz_path = os.path.join(self.raw_root_path, "amazon_electronics_{}.npz".format(self.dataset_name.replace("amazon-", "")))

        with np.load(npz_path) as data:

            x = sp.csr_matrix((data["attr_data"], data["attr_indices"], data["attr_indptr"]), data["attr_shape"]).todense().astype(np.float32)
            x[x > 0.0] = 1.0

            adj = sp.csr_matrix((data["adj_data"], data["adj_indices"], data["adj_indptr"]), data["adj_shape"]).tocoo()
            edge_index = np.stack([adj.row, adj.col], axis=0).astype(np.int32)
            edge_index, _ = remove_self_loop_edge(edge_index)
            edge_index, _ = convert_edge_to_directed(edge_index, merge_modes="max")

            y = data["labels"].astype(np.int32)

        graph = Graph(x=x, edge_index=edge_index, y=y)

        return graph


class AmazonComputersDataset(AmazonElectronicsDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__("amazon-computers", dataset_root_path=dataset_root_path)


class AmazonPhotoDataset(AmazonElectronicsDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__("amazon-photo", dataset_root_path=dataset_root_path)
