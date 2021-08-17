# coding=utf-8
from multiprocessing import Pool

from tf_geometric.data.graph import Graph
from tf_geometric.data.dataset import DownloadableDataset
import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class ModelNetDataset(DownloadableDataset):
    def __init__(self, dataset_name, download_urls=None, dataset_root_path=None, num_processes=50):
        super().__init__(dataset_name, download_urls,
                         download_file_name="{}.zip".format(dataset_name),
                         cache_name="cache.p",
                         dataset_root_path=dataset_root_path)
        self.num_processes = num_processes

    def read_off(self, off_file_info):
        off_fpath, label_index = off_file_info
        with open(off_fpath, "r", encoding="utf-8") as f:
            # jump top line, usually "OFF\n"
            line = f.readline()

            # some files does not have "\n" after "OFF", and the number information is in the first line after "OFF"
            if line != "OFF\n":
                line = line[3:]
            else:
                # Usually, the number information is in the second line
                line = f.readline()

            # parse numbers
            num_nodes, num_faces, _ = [int(item) for item in line.split()]

            node_feature_matrix = []
            for _ in range(num_nodes):
                line = f.readline()
                node_features = [float(item) for item in line.split()]
                node_feature_matrix.append(node_features)

            triangles = []
            for _ in range(num_faces):
                line = f.readline()
                items = [int(item) for item in line.split()]
                num_face_nodes = items[0]

                # triangle
                if num_face_nodes == 3:
                    triangles.append(items[1:])
                # rectangle
                else:
                    triangles.append([items[1], items[2], items[3]])
                    triangles.append([items[1], items[3], items[4]])

        x = np.array(node_feature_matrix)
        triangles = np.array(triangles)
        edges = np.concatenate([
            triangles[:, :2],
            triangles[:, 1:],
            triangles[:, ::2]
        ], axis=0)
        row, col = edges[:, 0], edges[:, 1]
        row, col = np.concatenate([row, col], axis=0), np.concatenate([col, row], axis=0)

        adj = sp.csr_matrix((np.ones_like(row), (row, col)), shape=[num_nodes, num_nodes])
        adj.data[adj.data > 1] = 1

        adj = adj.tocoo()
        edge_index = np.stack([adj.row, adj.col], axis=0)

        graph = Graph(x=x, edge_index=edge_index, y=[label_index])

        return graph

    def process(self):
        data_dir = os.path.join(self.raw_root_path, self.dataset_name)
        sorted_sub_dir_names = sorted([sub_dir_name for sub_dir_name in os.listdir(data_dir)
                                       if os.path.isdir(os.path.join(data_dir, sub_dir_name))])
        label_names = sorted_sub_dir_names

        # label_name_index_dict = {label_name: label_index for label_index, label_name in enumerate(label_names)}

        train_graphs = []
        test_graphs = []

        for label_index, label_name in enumerate(label_names):
            print("processing {} ({}/{}):".format(label_name, label_index + 1, len(label_names)))

            label_path = os.path.join(data_dir, label_name)

            for split_index, split in enumerate(["train", "test"]):
                print("\treading {} graphs".format(split))

                split_graphs = train_graphs if split == "train" else test_graphs
                split_path = os.path.join(label_path, split)
                off_fpaths = [os.path.join(split_path, off_fname) for off_fname in os.listdir(split_path) if
                              off_fname != ".DS_Store"]

                pool_inputs = [[off_fpath, label_index] for off_fpath in off_fpaths]
                with Pool(processes=self.num_processes) as p:
                    with tqdm(total=len(off_fpaths)) as pbar:
                        for _, graph in enumerate(p.imap_unordered(self.read_off, pool_inputs)):
                            split_graphs.append(graph)
                            pbar.update()

                # split_path = os.path.join(label_path, split)
                # for off_fname in tqdm(os.listdir(split_path)):
                #     if off_fname == ".DS_Store":
                #         continue
                #     off_fpath = os.path.join(split_path, off_fname)
                #     graph = self.read_off(off_fpath, label_index)
                #     split_graphs.append(graph)

        return train_graphs, test_graphs, label_names


class ModelNet10Dataset(ModelNetDataset):
    def __init__(self, dataset_root_path=None, num_processes=50):
        super().__init__(
            dataset_name="ModelNet10",
            download_urls="http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
            dataset_root_path=dataset_root_path,
            num_processes=num_processes
        )


class ModelNet40Dataset(ModelNetDataset):
    def __init__(self, dataset_root_path=None, num_processes=50):
        super().__init__(
            dataset_name="ModelNet40",
            download_urls="http://modelnet.cs.princeton.edu/ModelNet40.zip",
            dataset_root_path=dataset_root_path,
            num_processes=num_processes
        )
