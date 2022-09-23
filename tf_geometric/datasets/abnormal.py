# coding=utf-8

import numpy as np
from tf_sparse import SparseMatrix
from tf_geometric.data.dataset import DownloadableDataset
import os
from scipy.io import loadmat


def _csc_to_edge_index(x):
    x = x.tocoo()
    return np.stack([x.row, x.col], axis=0)


def _csc_to_sparse_matrix(x):
    x = x.tocoo()
    index = np.stack([x.row, x.col], axis=0)
    value = x.data.astype(np.float64)
    return SparseMatrix(index, value, shape=x.shape)


class _BaseAbnormalMATDataset(DownloadableDataset):
    def __init__(self, dataset_name, dataset_root_path=None):
        super().__init__(dataset_name,
                         download_urls=["https://github.com/CrawlScript/gnn_datasets/raw/master/Abnormal/{}.zip".format(
                             dataset_name)],
                         download_file_name="{}.zip".format(dataset_name),
                         cache_name=None, dataset_root_path=dataset_root_path)

    def process(self):
        mat_path = os.path.join(self.raw_root_path, "{}.mat".format(self.dataset_name))
        data = loadmat(mat_path)

        # x = _csc_to_sparse_matrix(data["features"])
        x = data["features"].tocoo().astype(np.float64)
        y = data["label"][0].astype(np.int64)

        edge_index_dict = {}

        for key, value in data.items():
            if key.startswith("net_") or key == "homo":
                edge_index = _csc_to_edge_index(value)
                edge_index_dict[key] = edge_index

        return x, edge_index_dict, y


class FDYelpChiDataset(_BaseAbnormalMATDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__("fd_yelp_chi", dataset_root_path)


class FDAmazonDataset(_BaseAbnormalMATDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__("fd_amazon", dataset_root_path)
