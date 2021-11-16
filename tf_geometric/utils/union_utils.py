# coding=utf-8
import tensorflow as tf
import numpy as np
import tf_sparse as tfs

def convert_union_to_numpy(data, dtype=None):
    if data is None:
        return data

    if tf.is_tensor(data):
        np_data = data.numpy()
    elif isinstance(data, list):
        np_data = np.array(data)
    else:
        np_data = data

    if dtype is not None:
        np_data = np_data.astype(dtype)

    return np_data


def union_len(data):
    if tf.is_tensor(data):
        return tfs.shape(data)[0]
    else:
        return len(data)