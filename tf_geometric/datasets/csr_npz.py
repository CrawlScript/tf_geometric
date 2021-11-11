# coding=utf-8
import numpy as np
from tf_geometric.data.graph import Graph
import scipy.sparse as sp
import os

from tf_geometric.utils.graph_utils import convert_edge_to_directed, remove_self_loop_edge
from tf_geometric.data.dataset import DownloadableDataset


class CSRNPZDataset(DownloadableDataset):

    def process(self):

        # npz_path = os.path.join(self.raw_root_path, "amazon_electronics_{}.npz".format(self.dataset_name.replace("amazon-", "")))
        npz_name = [fname for fname in os.listdir(self.raw_root_path) if fname.endswith(".npz")][0]
        npz_path = os.path.join(self.raw_root_path, npz_name)

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
