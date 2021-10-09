# coding=utf-8
import os

from tensorflow.python.keras.utils.data_utils import _extract_archive
from tf_geometric.utils.data_utils import download_file, load_cache, save_cache
from shutil import copy

DEFAULT_DATASETS_ROOT = "data"


def get_dataset_root_path(dataset_root_path=None, dataset_name=None,
                          datasets_root_path=DEFAULT_DATASETS_ROOT, mkdir=False):
    if dataset_root_path is None:
        dataset_root_path = os.path.join(datasets_root_path, dataset_name)
    dataset_root_path = os.path.abspath(dataset_root_path)

    if mkdir:
        os.makedirs(dataset_root_path, exist_ok=True)
    return dataset_root_path


class Dataset(object):
    pass


class DownloadableDataset(object):

    def __init__(self,
                 dataset_name,
                 download_urls=None,
                 download_file_name=None,
                 cache_name="cache.p",
                 dataset_root_path=None
                 ):
        self.dataset_name = dataset_name
        self.dataset_root_path = get_dataset_root_path(dataset_root_path, dataset_name)
        self.download_urls = download_urls
        self.download_file_name = download_file_name

        self.download_root_path = os.path.join(self.dataset_root_path, "download")
        self.raw_root_path = os.path.join(self.dataset_root_path, "raw")
        self.processed_root_path = os.path.join(self.dataset_root_path, "processed")

        if download_urls is not None:
            if download_file_name is None:
                download_file_name = "{}.zip".format(dataset_name)
            self.download_file_path = os.path.join(self.download_root_path, download_file_name)
        else:
            self.download_file_path = None

        self.cache_path = None if cache_name is None else os.path.join(self.processed_root_path, cache_name)

        self.build_dirs()

    @property
    def cache_enabled(self):
        return self.cache_path is not None

    def build_dirs(self):
        os.makedirs(self.download_root_path, exist_ok=True)
        os.makedirs(self.raw_root_path, exist_ok=True)
        os.makedirs(self.processed_root_path, exist_ok=True)

    def download(self):
        download_file(self.download_file_path, self.download_urls)

    def extract_raw(self):
        if len(os.listdir(self.raw_root_path)) == 0:
            if self.download_file_path.endswith(".npz"):
                copy(self.download_file_path, os.path.join(self.raw_root_path, self.download_file_name))
            else:
                _extract_archive(self.download_file_path, self.raw_root_path, archive_format="auto")
        else:
            print("raw data exists: {}, ignore".format(self.raw_root_path))

    def process(self):
        pass

    def load_data(self):
        if self.cache_enabled and os.path.exists(self.cache_path):
            print("cache file exists: {}, read cache".format(self.cache_path))
            return load_cache(self.cache_path)

        if self.download_urls is not None:
            self.download()
            self.extract_raw()
        else:
            print("downloading and extraction are ignored due to None download_urls")

        processed = self.process()

        if self.cache_enabled:
            print("save processed data to cache: ", self.cache_path)
            save_cache(processed, self.cache_path)

        return processed




















