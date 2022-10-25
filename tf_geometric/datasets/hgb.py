# coding=utf-8
import os.path
import json
from collections import defaultdict
import numpy as np
from tf_geometric.data.graph import HeteroDictGraph
from tf_geometric.data.dataset import DownloadableDataset


class HGBDataset(DownloadableDataset):

    def __init__(self, dataset_name, dataset_root_path=None):
        """

        :param dataset_name: "hgb_acm" | "hgb_dblp" | "hgb_freebase" | "hgb_imdb"
        :param dataset_root_path:
        """
        self.sub_dataset_name = dataset_name.split("_")[1]

        super().__init__(dataset_name=dataset_name,
                         download_urls=[
                             "https://github.com/CrawlScript/gnn_datasets/raw/master/hgb/{}.zip".format(self.sub_dataset_name)
                         ],
                         download_file_name="{}.zip".format(self.sub_dataset_name),
                         cache_name=None,
                         dataset_root_path=dataset_root_path,
                         )

    # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/hgb_dataset.py
    def process(self):
        # data = HeteroData()

        # node_types = {0: 'paper', 1, 'author', ...}
        # edge_types = {0: ('paper', 'cite', 'paper'), ...}

        data_dir = os.path.join(self.raw_root_path, self.sub_dataset_name)

        if self.sub_dataset_name in ['acm', 'dblp', 'imdb']:

            # with open(self.raw_paths[0], 'r') as f:  # `info.dat`
            with open(os.path.join(data_dir, "info.dat"), 'r') as f:  # `info.dat`
                info = json.load(f)
            n_types = info['node.dat']['node type']
            n_types = {int(k): v for k, v in n_types.items()}
            e_types = info['link.dat']['link type']
            e_types = {int(k): tuple(v.values()) for k, v in e_types.items()}
            for key, (src, dst, rel) in e_types.items():
                src, dst = n_types[int(src)], n_types[int(dst)]
                rel = rel.split('-')[1]
                rel = rel if rel != dst and rel[1:] != dst else 'to'
                e_types[key] = (src, rel, dst)
            num_classes = len(info['label.dat']['node type']['0'])
        elif self.sub_dataset_name in ['freebase']:
            # with open(self.raw_paths[0], 'r') as f:  # `info.dat`
            with open(os.path.join(data_dir, "info.dat"), 'r') as f:  # `info.dat`
                info = f.read().split('\n')
            start = info.index('TYPE\tMEANING') + 1
            end = info[start:].index('')
            n_types = [v.split('\t\t') for v in info[start:start + end]]
            n_types = {int(k): v.lower() for k, v in n_types}

            e_types = {}
            start = info.index('LINK\tSTART\tEND\tMEANING') + 1
            end = info[start:].index('')
            for key, row in enumerate(info[start:start + end]):
                row = row.split('\t')[1:]
                src, dst, rel = [v for v in row if v != '']
                src, dst = n_types[int(src)], n_types[int(dst)]
                rel = rel.split('-')[1]
                e_types[key] = (src, rel, dst)
        else:  # Link prediction:
            raise NotImplementedError

        # Extract node information:
        mapping_dict = {}  # Maps global node indices to local ones.
        x_dict = defaultdict(list)
        num_nodes_dict = defaultdict(lambda: 0)

        # with open(self.raw_paths[1], 'r') as f:  # `node.dat`
        with open(os.path.join(data_dir, "node.dat"), 'r') as f:  # `node.dat
            xs = [v.split('\t') for v in f.read().split('\n')[:-1]]
        for x in xs:
            n_id, n_type = int(x[0]), n_types[int(x[2])]
            mapping_dict[n_id] = num_nodes_dict[n_type]
            num_nodes_dict[n_type] += 1
            if len(x) >= 4:  # Extract features (in case they are given).
                x_dict[n_type].append([float(v) for v in x[3].split(',')])
            else:
                x_dict[n_type].append([np.inf])


        x_dict = {ntype: np.array(x, dtype=np.float64) for ntype, x in x_dict.items()}


        # for n_type in n_types.values():
        #     if len(x_dict[n_type]) == 0:
        #         data[n_type].num_nodes = num_nodes_dict[n_type]
        #     else:
        #         data[n_type].x = torch.tensor(x_dict[n_type])

        # edge_index_dict = defaultdict(list)
        # edge_weight_dict = defaultdict(list)

        edge_dict = defaultdict(list)
        edge_weight_dict = defaultdict(list)

        # with open(self.raw_paths[2], 'r') as f:  # `link.dat`
        with open(os.path.join(data_dir, "link.dat"), 'r') as f:  # `link.dat`
            edges = [v.split('\t') for v in f.read().split('\n')[:-1]]
        for src, dst, rel, weight in edges:
            e_type = e_types[int(rel)]
            src, dst = mapping_dict[int(src)], mapping_dict[int(dst)]
            edge_dict[e_type].append([src, dst])
            edge_weight_dict[e_type].append(float(weight))

        edge_index_dict = {e_type: np.array(edges, dtype=np.int64).T for e_type, edges in edge_dict.items()}
        edge_weight_dict = {e_type: np.array(edge_weight, dtype=np.float64)
                            for e_type, edge_weight in edge_weight_dict.items()
                            if not np.allclose(edge_weight, np.ones_like(edge_weight))}

        # for e_type in e_types.values():
        #     edge_index = torch.tensor(edge_index_dict[e_type])
        #     edge_weight = torch.tensor(edge_weight_dict[e_type])
        #     data[e_type].edge_index = edge_index.t().contiguous()
        #     # Only add "weighted" edgel to the graph:
        #     if not torch.allclose(edge_weight, torch.ones_like(edge_weight)):
        #         data[e_type].edge_weight = edge_weight

        y_dict = {}
        train_mask_dict = {}
        test_mask_dict = {}


        # with open(self.raw_paths[3], 'r') as f:  # `label.dat`
        with open(os.path.join(data_dir, "label.dat"), 'r') as f:  # `label.dat`
            train_ys = [v.split('\t') for v in f.read().split('\n')[:-1]]
        # with open(self.raw_paths[4], 'r') as f:  # `label.dat.test`
        with open(os.path.join(data_dir, "label.dat.test"), 'r') as f:  # `label.dat.test`
            test_ys = [v.split('\t') for v in f.read().split('\n')[:-1]]

        for y in train_ys:
            n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])]

            if n_type not in y_dict:

                num_nodes = x_dict[n_type].shape[0]# if n_type in x_dict else num_nodes_dict[n_type]

                if self.sub_dataset_name in ['imdb']:  # multi-label
                    y_dict[n_type] = np.zeros([num_nodes, num_classes], dtype=np.int64)
                else:
                    y_dict[n_type] = np.full([num_nodes], -1, dtype=np.int64)

                train_mask_dict[n_type] = np.zeros(num_nodes).astype(np.bool)
                test_mask_dict[n_type] = np.zeros(num_nodes).astype(np.bool)


            # if not hasattr(data[n_type], 'y'):
            #     num_nodes = data[n_type].num_nodes
            #     if self.name in ['imdb']:  # multi-label
            #         data[n_type].y = torch.zeros((num_nodes, num_classes))
            #     else:
            #         data[n_type].y = torch.full((num_nodes,), -1).long()
            #     data[n_type].train_mask = torch.zeros(num_nodes).bool()
            #     data[n_type].test_mask = torch.zeros(num_nodes).bool()

            if y_dict[n_type].ndim > 1:  # multi-label
                for v in y[3].split(','):
                    y_dict[n_type][n_id, int(v)] = 1
            else:
                y_dict[n_type][n_id] = int(y[3])
            train_mask_dict[n_type][n_id] = True

            # if data[n_type].y.dim() > 1:  # multi-label
            #     for v in y[3].split(','):
            #         data[n_type].y[n_id, int(v)] = 1
            # else:
            #     data[n_type].y[n_id] = int(y[3])
            # data[n_type].train_mask[n_id] = True

        for y in test_ys:
            n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])]
            if y_dict[n_type].ndim > 1:  # multi-label
                for v in y[3].split(','):
                    y_dict[n_type][n_id, int(v)] = 1
            else:
                y_dict[n_type][n_id] = int(y[3])
            test_mask_dict[n_type][n_id] = True


        # for y in test_ys:
        #     n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])]
        #     if data[n_type].y.dim() > 1:  # multi-label
        #         for v in y[3].split(','):
        #             data[n_type].y[n_id, int(v)] = 1
        #     else:
        #         data[n_type].y[n_id] = int(y[3])
        #     data[n_type].test_mask[n_id] = True


        hetero_graph = HeteroDictGraph(x_dict=x_dict,
                                       edge_index_dict=edge_index_dict,
                                       y_dict=y_dict,
                                       edge_weight_dict=edge_weight_dict)

        return hetero_graph, train_mask_dict, test_mask_dict


class HGBACMDataset(HGBDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__("hgb_acm", dataset_root_path)


class HGBDBLPDataset(HGBDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__("hgb_dblp", dataset_root_path)


class HGBFreebaseDataset(HGBDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__("hgb_freebase", dataset_root_path)


class HGBIMDBDataset(HGBDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__("hgb_imdb", dataset_root_path)
