# coding=utf-8

import numpy as np
import os

import networkx as nx
from tf_geometric.data.graph import Graph
from tf_geometric.data.dataset import DownloadableDataset
import json

from tf_geometric.utils.data_utils import load_cache
from tf_geometric.utils.graph_utils import convert_edge_to_directed


class PPIDataset(DownloadableDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="PPI",
                         download_urls=[
                             "https://data.dgl.ai/dataset/ppi.zip",
                             "https://github.com/CrawlScript/gnn_datasets/raw/master/PPI/ppi.zip"
                         ],
                         download_file_name="ppi.zip",
                         cache_name="cache.p",
                         dataset_root_path=dataset_root_path,
                         )

    def process(self):

        splits = ["train", "valid", "test"]

        split_data_dict = {
            split: [] for split in splits
        }

        for split in split_data_dict.keys():
            split_graph_ids = np.load(os.path.join(self.raw_root_path, "{}_graph_id.npy".format(split)))
            split_features = np.load(os.path.join(self.raw_root_path, "{}_feats.npy".format(split))).astype(np.float32)
            split_labels = np.load(os.path.join(self.raw_root_path, "{}_labels.npy".format(split))).astype(np.int32)

            nx_graph_path = os.path.join(self.raw_root_path, "{}_graph.json".format(split))
            with open(nx_graph_path, "r", encoding="utf-8") as f:
                nx_graph = nx.DiGraph(nx.json_graph.node_link_graph(json.load(f)))

            split_unique_graph_ids = sorted(set(split_graph_ids))

            for graph_id in split_unique_graph_ids:
                mask_indices = np.where(split_graph_ids == graph_id)[0]

                min_node_index = np.min(mask_indices)

                edge_index = nx_graph.subgraph(mask_indices).edges
                edge_index = np.array(edge_index).T - min_node_index

                edge_index, _ = convert_edge_to_directed(edge_index)

                graph = Graph(
                    x=split_features[mask_indices],
                    edge_index=edge_index,
                    y=split_labels[mask_indices]
                )
                split_data_dict[split].append(graph)
                # print("split: ", split)

        processed_data = [split_data_dict[split] for split in splits]
        return processed_data
