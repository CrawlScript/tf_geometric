# coding=utf-8
from tensorflow import keras
import pickle



def download_file(path, url_or_urls):
    if not isinstance(url_or_urls, list):
        urls = [url_or_urls]
    else:
        urls = url_or_urls

    last_except = None
    for url in urls:
        try:
            return keras.utils.get_file(path, origin=url)
        except Exception as e:
            last_except = e
            print(e)

    raise last_except


def save_cache(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_cache(path):
    # if not os.path.exists(path):
    #     return None

    with open(path, "rb") as f:
        return pickle.load(f)