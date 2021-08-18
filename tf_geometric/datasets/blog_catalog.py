# coding=utf-8

import numpy as np
from tf_geometric.data.dataset import DownloadableDataset
import os
import scipy.io as scio


class MultiLabelBlogCatalogDataset(DownloadableDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="MultiLabelBlogCatalog",
                         download_urls=[
                             "https://github.com/CrawlScript/gnn_datasets/raw/master/BlogCatalog/multi_label_blog_catalog.zip",
                             "http://cdn.zhuanzhi.ai/github/multi_label_blog_catalog.zip"
                         ],
                         download_file_name="multi_label_blog_catalog.zip",
                         cache_name="cache.p",
                         dataset_root_path=dataset_root_path,
                         )

    def process(self):
        data_path = os.path.join(self.raw_root_path, "multi_label_blog_catalog.mat")
        data = scio.loadmat(data_path)

        adj = data['network'].tocoo()
        edge_index = np.stack([adj.row, adj.col], axis=0)

        y = data['group'].tocoo().toarray().astype(np.float32)

        return edge_index, y
