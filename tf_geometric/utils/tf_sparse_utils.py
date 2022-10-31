# coding=utf-8

import tensorflow as tf
import tf_sparse as tfs
import numpy as np


def sparse_tensor_gather_sub(sparse_tensor, sub_index, axis=0):
    gather_index = sparse_tensor.indices[:, axis]
    if axis in [0, -2]:
        other_axis = 1
    else:
        other_axis = 0

    dense_shape = tf.shape(sparse_tensor)

    index_mask = tf.scatter_nd(tf.expand_dims(sub_index, axis=-1), tf.ones_like(sub_index), [dense_shape[axis]])
    index_mask = tf.cast(index_mask, tf.bool)

    gather_mask = tf.gather(index_mask, gather_index)

    masked_values = tf.boolean_mask(sparse_tensor.values, gather_mask)
    masked_indices = tf.boolean_mask(sparse_tensor.indices, gather_mask)

    reverse_index = tf.cast(tf.fill([tf.reduce_max(sub_index) + 1], -1), tf.int32)
    reverse_index = tf.tensor_scatter_nd_update(reverse_index, tf.expand_dims(sub_index, axis=-1),
                                                tf.range(tf.shape(sub_index)[0]))

    masked_gather_index = masked_indices[:, axis]
    masked_gather_index = tf.gather(reverse_index, masked_gather_index)
    masked_gather_index = tf.cast(masked_gather_index, dtype=tf.int64)

    masked_other_index = masked_indices[:, other_axis]

    new_indices = [None, None]
    new_indices[axis] = masked_gather_index
    new_indices[other_axis] = masked_other_index

    new_shape = [None, None]
    new_shape[axis] = tf.shape(sub_index)[0]
    new_shape[other_axis] = dense_shape[other_axis]
    new_shape = tf.cast(new_shape, tf.int64)

    new_indices = tf.stack(new_indices, axis=1)

    new_sparse_tensor = tf.sparse.SparseTensor(
        indices=new_indices,
        values=masked_values,
        dense_shape=new_shape
    )
    new_sparse_tensor = tf.sparse.reorder(new_sparse_tensor)
    return new_sparse_tensor


def sparse_gather_sub(x, sub_index, axis=0):
    is_sparse_matrix = isinstance(x, tfs.SparseMatrix)

    if is_sparse_matrix:
        sparse_tensor = x.to_sparse_tensor()
    else:
        sparse_tensor = x

    output_sparse_tensor = sparse_tensor_gather_sub(sparse_tensor, sub_index, axis)

    if is_sparse_matrix:
        return x.__class__.from_sparse_tensor(output_sparse_tensor)
    else:
        return output_sparse_tensor


def compute_num_or_size_splits(num_h_features, num_splits):

    if num_splits is None or num_splits == 1:
        num_or_size_splits = None
    elif num_h_features % num_splits == 0:
        num_or_size_splits = num_splits
    else:
        split_size = np.ceil(num_h_features / num_splits).astype(np.int64)

        num_pre_splits = np.floor(num_h_features / split_size).astype(np.int64)
        last_split_size = num_h_features % split_size

        num_or_size_splits = [split_size] * num_pre_splits + ([last_split_size] if last_split_size > 0 else [])

        if len(num_or_size_splits) != num_splits:
            raise Exception(
                "cannot split H of shape [None, {}] into {} matrices, please provide a valid num_splits"
                .format(num_h_features, num_splits))

    return num_or_size_splits
