# coding=utf-8

import numpy as np
import os
from tqdm import tqdm
import networkx as nx
from tf_geometric.data.graph import Graph
from tf_geometric.data.dataset import DownloadableDataset
import json

from tf_geometric.utils.data_utils import load_cache
from tf_geometric.utils.graph_utils import convert_edge_to_directed


class TUDataset(DownloadableDataset):

    def __init__(self, dataset_name, dataset_root_path=None):
        tu_base_url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'
        super().__init__(dataset_name=dataset_name,
                         download_urls=[
                             "{}/{}.zip".format(tu_base_url, dataset_name)
                         ],
                         download_file_name="{}.zip".format(dataset_name),
                         cache_name="{}.p".format(dataset_name),
                         dataset_root_path=dataset_root_path,
                         )
        self.txt_root_path = os.path.join(self.raw_root_path, self.dataset_name)
        self.prefix = "{}_".format(self.dataset_name)

    def _build_label_id_index_dict(self, label_ids):
        sorted_label_ids = sorted(list(set(label_ids)))
        label_id_index_dict = {label_id: label_index for label_index, label_id in enumerate(sorted_label_ids)}
        return label_id_index_dict

    def _convert_label_ids_to_indices(self, label_ids):
        label_id_index_dict = self._build_label_id_index_dict(label_ids)
        return np.array([label_id_index_dict[label_id] for label_id in label_ids])

    def process(self):

        print("reading graph_indicator...")
        node_graph_index = self.read_txt_as_array("graph_indicator", dtype=np.int32)
        graph_index_offset = node_graph_index.min()
        node_graph_index -= graph_index_offset
        print("reading edges...")
        edges = self.read_txt_as_array("A", dtype=np.int32) - graph_index_offset
        edge_graph_index = node_graph_index[edges[:, 0]]
        num_graphs = node_graph_index.max() + 1

        # node_labels
        node_label_ids = self.read_txt_as_array("node_labels", dtype=np.int32)
        if node_label_ids is not None:
            node_labels = self._convert_label_ids_to_indices(node_label_ids)
        else:
            node_labels = None

        # node_labels
        edge_label_ids = self.read_txt_as_array("edge_labels", dtype=np.int32)
        if edge_label_ids is not None:
            edge_labels = self._convert_label_ids_to_indices(edge_label_ids)
        else:
            edge_labels = None

        # node_labels
        node_attributes_list = self.read_txt_as_array("node_attributes", dtype=np.float32)

        # graph_labels
        graph_label_ids = self.read_txt_as_array("graph_labels", dtype=np.int32)
        if graph_label_ids is not None:
            graph_labels = self._convert_label_ids_to_indices(graph_label_ids)
        else:
            graph_labels = None

        def create_empty_graph():
            graph = {"edge_index": []}

            if node_labels is not None:
                graph["node_labels"] = []

            if node_attributes_list is not None:
                graph["node_attributes"] = []

            if edge_labels is not None:
                graph["edge_labels"] = []

            if graph_labels is not None:
                graph["graph_label"] = None

            return graph

        graphs = [create_empty_graph() for _ in range(num_graphs)]

        start_node_index = np.full([num_graphs], -1).astype(np.int32)
        for node_index, graph_index in enumerate(node_graph_index):
            if start_node_index[graph_index] < 0:
                start_node_index[graph_index] = node_index

        for graph_index, graph in enumerate(graphs):
            end_index = start_node_index[graph_index + 1] if graph_index < num_graphs - 1 else len(node_graph_index)
            num_nodes = end_index - start_node_index[graph_index]
            graph["num_nodes"] = num_nodes

        # edge_index
        for graph_index, edge in zip(edge_graph_index, edges):
            graph = graphs[graph_index]
            graph["edge_index"].append(edge)

        if node_labels is not None:
            # node_labels -= node_labels.min()
            for graph_index, node_label in zip(node_graph_index, node_labels):
                graph = graphs[graph_index]
                graph["node_labels"].append(node_label)

        if edge_labels is not None:
            # edge_labels -= edge_labels.min()
            for graph_index, edge_label in zip(edge_graph_index, edge_labels):
                graph = graphs[graph_index]
                graph["edge_labels"].append(edge_label)


        if node_attributes_list is not None:
            node_attributes_list = node_attributes_list.reshape(node_attributes_list.shape[0], -1)
            for graph_index, node_attributes in zip(node_graph_index, node_attributes_list):
                graph = graphs[graph_index]
                graph["node_attributes"].append(node_attributes)


        # graph_labels

        if graph_labels is not None:
            # graph_labels -= graph_labels.min()
            for graph, graph_label in zip(graphs, graph_labels):
                graph["graph_label"] = np.array([graph_label]).astype(np.int32)

        for i, graph in enumerate(graphs):
            edge_index = np.array(graph["edge_index"]).T - start_node_index[i]
            graph["edge_index"] = edge_index

            if node_labels is not None:
                graph["node_labels"] = np.array(graph["node_labels"]).astype(np.int32)

            if node_attributes_list is not None:
                graph["node_attributes"] = np.array(graph["node_attributes"]).astype(np.float32)

            if edge_labels is not None:
                graph["edge_labels"] = np.array(graph["edge_labels"]).astype(np.int32)

            num_nodes = graph["num_nodes"]
            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(np.arange(num_nodes))
            nx_graph.add_edges_from(edge_index.T)
            degrees = np.array([nx_graph.degree(node_index) for node_index in range(num_nodes)]).astype(np.int32)
            graph["degrees"] = degrees

        return graphs


    def get_path_by_fid(self, fid):
        fname = "{}_{}.txt".format(self.dataset_name, fid)
        return os.path.join(self.txt_root_path, fname)


    def read_txt_as_array(self, fid, dtype):
        path = self.get_path_by_fid(fid)
        if not os.path.exists(path):
            return None
        data_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split(",")
                items = [dtype(item) for item in items]
                data = items[0] if len(items) == 1 else items
                data_list.append(data)
        data_list = np.array(data_list).astype(dtype)
        return data_list




