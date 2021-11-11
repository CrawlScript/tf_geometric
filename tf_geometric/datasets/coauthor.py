# coding=utf-8
from tf_geometric.datasets.csr_npz import CSRNPZDataset


class CoauthorDataset(CSRNPZDataset):

    def __init__(self, dataset_name, dataset_root_path=None):
        """

        :param dataset_name: "coauthor-cs" | "coauthor-physics"
        :param dataset_root_path:
        """
        super().__init__(dataset_name=dataset_name,
                         download_urls=[
                             "https://github.com/CrawlScript/gnn_datasets/raw/master/Coauthor/{}.zip".format(dataset_name),
                             "http://cdn.zhuanzhi.ai/github/{}.zip".format(dataset_name)
                         ],
                         download_file_name="{}.zip".format(dataset_name),
                         cache_name=None,
                         dataset_root_path=dataset_root_path,
                         )


class CoauthorCSDataset(CoauthorDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__("coauthor-cs", dataset_root_path=dataset_root_path)


class CoauthorPhysicsDataset(CoauthorDataset):

    def __init__(self, dataset_root_path=None):
        super().__init__("coauthor-physics", dataset_root_path=dataset_root_path)
