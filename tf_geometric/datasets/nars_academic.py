# coding=utf-8
import os.path
import json
from collections import defaultdict
import numpy as np
from scipy.io import loadmat

from tf_geometric.data.graph import HeteroDictGraph
from tf_geometric.data.dataset import DownloadableDataset


class _NARSAcademicDataset(DownloadableDataset):

    def __init__(self, dataset_name, dataset_root_path=None):
        """

        :param dataset_name: "nars_academic_acm"
        :param dataset_root_path:
        """
        self.sub_dataset_name = dataset_name.split("_")[-1]

        super().__init__(dataset_name=dataset_name,
                         download_urls=[
                             "https://github.com/CrawlScript/gnn_datasets/raw/master/nars_academic/{}.zip".format(self.sub_dataset_name)
                         ],
                         download_file_name="{}.zip".format(self.sub_dataset_name),
                         cache_name=None,
                         dataset_root_path=dataset_root_path,
                         )

    # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/hgb_dataset.py
    def process(self):
        from scipy import io as sio

        data = loadmat(os.path.join(self.raw_root_path, "acm.mat"))
        p_vs_l = data['PvsL']  # paper-field?
        p_vs_a = data['PvsA']  # paper-author
        p_vs_t = data['PvsT']  # paper-term, bag of words
        p_vs_c = data['PvsC']  # paper-conference, labels come from that

        # We assign
        # (1) KDD papers as class 0 (data mining),
        # (2) SIGMOD and VLDB papers as class 1 (database),
        # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
        p_vs_l = p_vs_l[p_selected]
        p_vs_a = p_vs_a[p_selected]
        p_vs_t = p_vs_t[p_selected]
        p_vs_c = p_vs_c[p_selected]

        # pa = dgl.bipartite(p_vs_a, 'paper', 'pa', 'author')
        # pl = dgl.bipartite(p_vs_l, 'paper', 'pf', 'field')

        p_vs_a = p_vs_a.tocoo()
        p_vs_l = p_vs_l.tocoo()

        # pa = dgl.heterograph({('paper', 'pa', 'author'): (p_vs_a.col, p_vs_a.row)})
        # pl = dgl.heterograph({('paper', 'pf', 'field'): (p_vs_l.col, p_vs_l.row)})

        # gs = [pa, pl]
        # hg = dgl.hetero_from_relations(gs)
        # hg = dgl.heterograph({
        #     ('paper', 'pa', 'author'): (p_vs_a.row, p_vs_a.col),
        #     ('paper', 'pf', 'field'): (p_vs_l.row, p_vs_l.col)
        # })



        edge_index_dict = {
            ('paper', 'pa', 'author'): np.stack([p_vs_a.row, p_vs_a.col], axis=0).astype(np.int64),
            ('paper', 'pf', 'field'): np.stack([p_vs_l.row, p_vs_l.col], axis=0).astype(np.int64)
        }

        # features = torch.FloatTensor(p_vs_t.toarray())
        # features = p_vs_t.toarray().astype(np.float64)

        num_authors = p_vs_a.col.max() + 1
        num_fields = p_vs_l.col.max() + 1

        x_dict = {
            "paper": p_vs_t.toarray().astype(np.float64),
            "author": np.zeros([num_authors, 1], dtype=np.float32),
            "field": np.zeros([num_fields, 1], dtype=np.float32),
        }

        pc_p, pc_c = p_vs_c.nonzero()
        labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            labels[pc_p[pc_c == conf_id]] = label_id

        # labels = torch.LongTensor(labels)
        labels = np.array(labels, dtype=np.int64)

        y_dict = {"paper": labels}

        # num_classes = 3

        float_mask = np.zeros(len(pc_p))
        for conf_id in conf_ids:
            pc_c_mask = (pc_c == conf_id)
            float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
        train_index = np.where(float_mask <= 0.2)[0]
        valid_index = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
        test_index = np.where(float_mask > 0.3)[0]

        # hg.nodes["paper"].data["feat"] = features

        hetero_graph = HeteroDictGraph(x_dict=x_dict, edge_index_dict=edge_index_dict, y_dict=y_dict)

        target_node_type = "paper"

        return hetero_graph, target_node_type, (train_index, valid_index, test_index)

        # return hg, labels, num_classes, train_idx, val_idx, test_idx


class NARSACMDataset(_NARSAcademicDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__("nars_academic_acm", dataset_root_path)

