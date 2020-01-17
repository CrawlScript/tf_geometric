# coding=utf-8

import numpy as np
import os

import networkx as nx
from tf_geometric import Graph
from tf_geometric.data.dataset import DownloadableDataset
import json

from tf_geometric.utils.data_utils import load_cache


class PPIDataset(DownloadableDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="PPI",
                         download_urls=[
                             "https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip",
                             "https://github.com/CrawlScript/gnn_datasets/raw/master/PPI/ppi.zip"
                         ],
                         download_file_name="ppi.zip",
                         cache_name="cache.p",
                         dataset_root_path=dataset_root_path,
                         )

    def process(self):

        processed_data = {
            "train": [],
            "valid": [],
            "test": []
        }

        for split in processed_data.keys():
            split_graph_ids = np.load(os.path.join(self.raw_root_path, "{}_graph_id.npy".format(split)))
            split_features = np.load(os.path.join(self.raw_root_path, "{}_feats.npy".format(split)))
            split_labels = np.load(os.path.join(self.raw_root_path, "{}_labels.npy".format(split)))

            nx_graph_path = os.path.join(self.raw_root_path, "{}_graph.json".format(split))
            with open(nx_graph_path, "r", encoding="utf-8") as f:
                nx_graph = nx.DiGraph(nx.json_graph.node_link_graph(json.load(f)))

            split_unique_graph_ids = sorted(set(split_graph_ids))

            for graph_id in split_unique_graph_ids:
                mask_indices = np.where(split_graph_ids == graph_id)[0]

                min_node_index = np.min(mask_indices)

                edge_index = nx_graph.subgraph(mask_indices).edges
                edge_index = np.array(edge_index).T - min_node_index

                # use upper adj matrix
                row, col = edge_index
                upper_mask = row < col
                edge_index = edge_index[:, upper_mask]
                graph = Graph(
                    x=split_features[mask_indices],
                    edge_index=edge_index,
                    y=split_labels[mask_indices],
                    directed=False
                )
                processed_data[split].append(graph)
            return processed_data
