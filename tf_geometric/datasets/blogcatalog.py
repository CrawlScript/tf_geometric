# coding=utf-8
import tensorflow as tf
import numpy as np
from tf_geometric.data.graph import Graph
from tf_geometric.data.dataset import DownloadableDataset
import os
import scipy.io as scio

class Blogcatalog(DownloadableDataset):

    def __init__(self, dataset_name, task="semi_supervised", dataset_root_path=None):
        """
        :param dataset_name: "blogcatalog"
        :param task: "semi_supervised" | "supervised"
        :param dataset_root_path:
        """
        valid_tasks = ["semi_supervised"]
        if task not in valid_tasks:
            raise Exception("invalid task name for blogcatalog dataset: \"{}\"\tvalid task names: {}".format(task, valid_tasks))
        self.task = task

        super().__init__(dataset_name=dataset_name,
                         download_urls=[
                             "https://github.com/CrawlScript/gnn_datasets/raw/master/BlogCatalog/{}.zip".format(
                                 dataset_name)
                         ],
                         download_file_name="{}.zip".format(dataset_name),
                         cache_name=None,
                         dataset_root_path=dataset_root_path,
                         )

    def process(self):
        data_name = 'blogcatalog.mat'
        data_path = os.path.join(self.raw_root_path, data_name)
        data = scio.loadmat(data_path)

        edge_index_coo = data['network'].tocoo()
        edge_index_row, edge_index_col = edge_index_coo.row, edge_index_coo.col
        edge_index = np.vstack((edge_index_row, edge_index_col))

        y = (data['group'].tocoo().toarray()).astype(np.float32)

        node_num = y.shape[0]
        x = tf.sparse.eye(node_num)

        graph = Graph(x=x, edge_index=edge_index, y=y)

        return graph

class BlogcatalogDataset(Blogcatalog):

    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="blogcatalog", dataset_root_path=dataset_root_path)


