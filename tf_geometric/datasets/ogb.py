# coding=utf-8

from tf_geometric.data.graph import Graph
from tf_geometric.data.dataset import DownloadableDataset
from tf_geometric.utils.graph_utils import convert_edge_to_directed
from ogb_lite.nodeproppred import NodePropPredDataset
import numpy as np


class OGBNodePropPredDataset(DownloadableDataset):
    """
    OGB Node Property Prediction Datasets: https://ogb.stanford.edu/docs/nodeprop/
    """

    def __init__(self, dataset_name, dataset_root_path=None):
        """
        OGB Node Property Prediction Datasets: https://ogb.stanford.edu/docs/nodeprop/

        :param dataset_name: "ogbn-arxiv" | "ogbn-products" | "ogbn-proteins" | "ogbn-papers100M" | "ogbn-mag"
        :param dataset_root_path:
        """

        super().__init__(dataset_name=dataset_name,
                         download_urls=None,
                         download_file_name=None,
                         cache_name="cache.p",
                         dataset_root_path=dataset_root_path,
                         )

    # https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    def process(self):
        dataset = NodePropPredDataset(name=self.dataset_name, root=self.download_root_path)

        graph, label = dataset[0]  # graph: library-agnostic graph object

        x = graph["node_feat"]
        edge_index = graph["edge_index"]

        # convert edge_index to directed
        edge_index, _ = convert_edge_to_directed(edge_index, None)

        label = label.flatten().astype(np.int32)
        graph = Graph(x=x, edge_index=edge_index, y=label)

        split_index = dataset.get_idx_split()
        train_index, valid_index, test_index = split_index["train"], split_index["valid"], split_index["test"]

        return graph, (train_index, valid_index, test_index)


