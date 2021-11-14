# coding=utf-8

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.linalg.sparse.sparse_csr_matrix_ops import sparse_tensor_to_csr_sparse_matrix, \
    sparse_matrix_sparse_mat_mul, csr_sparse_matrix_to_sparse_tensor

from tf_geometric.utils.graph_utils import add_self_loop_edge, merge_duplicated_edge, remove_self_loop_edge

from tf_geometric.data.graph import Graph

from tf_geometric.nn.kernel.segment import segment_softmax
import warnings

"""
Sparse Matrix for Computation
"""


class SparseMatrix(object):
    # https://stackoverflow.com/questions/40252765/overriding-other-rmul-with-your-classs-mul
    __array_priority__ = 10000

    def __init__(self, edge_index, edge_weight=None, shape=None, merge=True):
        """
        Sparse Matrix for efficient computation.
        :param edge_index:
        :param edge_weight:
        :param shape: [num_rows, num_cols], shape of the adjacency matrix.
        :param merge: Whether to merge duplicated edge
        """

        self.edge_index = Graph.cast_edge_index(edge_index)

        edge_index_is_tensor = tf.is_tensor(edge_index)

        if edge_weight is not None:
            self.edge_weight = Graph.cast_edge_weight(edge_weight)
        else:
            if edge_index_is_tensor:
                num_edges = tf.shape(self.edge_index)[1]
                edge_weight = tf.ones([num_edges], dtype=tf.float32)
            else:
                num_edges = np.shape(self.edge_index)[1]
                edge_weight = np.ones([num_edges], dtype=np.float32)
            self.edge_weight = edge_weight

        if merge:
            self.edge_index, [self.edge_weight] = merge_duplicated_edge(self.edge_index, [self.edge_weight],
                                                                        merge_modes=["sum"])

        if shape is None:
            if edge_index_is_tensor:
                num_nodes = tf.reduce_max(edge_index) + 1
            else:
                num_nodes = np.max(edge_index) + 1
            self.shape = [num_nodes, num_nodes]
        else:
            self.shape = shape

    @property
    def row(self):
        return self.edge_index[0]

    @property
    def col(self):
        return self.edge_index[1]

    def add_self_loop(self, fill_weight=1.0):
        num_nodes = self.shape[0]
        updated_edge_index, updated_edge_weight = add_self_loop_edge(self.edge_index, num_nodes,
                                                                     edge_weight=self.edge_weight,
                                                                     fill_weight=fill_weight)
        return SparseMatrix(updated_edge_index, updated_edge_weight, self.shape)

    def remove_self_loop(self):
        updated_edge_index, updated_edge_weight = remove_self_loop_edge(self.edge_index, edge_weight=self.edge_weight)
        return SparseMatrix(updated_edge_index, updated_edge_weight, self.shape)

    def negative(self):
        return SparseMatrix(
            edge_index=self.edge_index,
            edge_weight=-self.edge_weight,
            shape=self.shape
        )

    def __neg__(self):
        return self.negative()

    def transpose(self):
        row, col = self.edge_index[0], self.edge_index[1]
        transposed_edge_index = tf.stack([col, row], axis=0)
        return SparseMatrix(transposed_edge_index, edge_weight=self.edge_weight, shape=[self.shape[1], self.shape[0]])

    def _reduce(self, segment_func, axis=-1, keepdims=False):

        # reduce by row
        if axis == -1 or axis == 1:
            reduce_axis = 0
        # reduce by col
        elif axis == 0 or axis == -2:
            reduce_axis = 1
        else:
            raise Exception("Invalid axis value: {}, axis shoud be -1, -2, 0, or 1".format(axis))

        reduce_index = self.edge_index[reduce_axis]
        num_reduced = self.shape[reduce_axis]

        reduced_edge_weight = segment_func(self.edge_weight, reduce_index, num_reduced)
        if keepdims:
            reduced_edge_weight = tf.expand_dims(reduced_edge_weight, axis=axis)
        return reduced_edge_weight

    def reduce_sum(self, axis=-1, keepdims=False):
        return self._reduce(tf.math.unsorted_segment_sum, axis=axis, keepdims=keepdims)

    def reduce_min(self, axis=-1, keepdims=False):
        return self._reduce(tf.math.unsorted_segment_min, axis=axis, keepdims=keepdims)

    def _matmul_dense(self, h):
        row, col = self.edge_index[0], self.edge_index[1]
        repeated_h = tf.gather(h, col)
        if self.edge_weight is not None:
            repeated_h *= tf.expand_dims(self.edge_weight, axis=-1)
        reduced_h = tf.math.unsorted_segment_sum(repeated_h, row, num_segments=self.shape[0])
        return reduced_h

    def _matmul_sparse(self, other):

        warnings.warn("The operation \"SparseMatrix @ SparseMatrix\" does not support gradient computation.")

        csr_matrix_a = self._to_csr_sparse_matrix()
        csr_matrix_b = other._to_csr_sparse_matrix()

        csr_matrix_c = sparse_matrix_sparse_mat_mul(
            a=csr_matrix_a, b=csr_matrix_b, type=self.edge_weight.dtype
        )

        sparse_tensor_c = csr_sparse_matrix_to_sparse_tensor(
            csr_matrix_c, type=self.edge_weight.dtype
        )

        return SparseMatrix.from_sparse_tensor(sparse_tensor_c)

    # sparse_adj @ other
    def matmul(self, other):
        if isinstance(other, SparseMatrix):
            return self._matmul_sparse(other)
        else:
            return self._matmul_dense(other)

    # h @ sparse_adj
    def rmatmul_dense(self, h):
        # h'
        transposed_h = tf.transpose(h, [1, 0])
        # sparse_adj' @ h'
        transpoed_output = self.transpose() @ transposed_h
        # h @ sparse_adj = (sparse_adj' @ h')'
        output = tf.transpose(transpoed_output, [1, 0])
        return output

    # # other_sparse_adj @ sparse_adj
    # def rmatmul_sparse(self, other):
    #     # h'
    #     transposed_other = other.transpose()
    #     # sparse_adj' @ h'
    #     transpoed_output = self.transpose() @ transposed_other
    #     # h @ sparse_adj = (sparse_adj' @ h')'
    #     output = transpoed_output.transpose()
    #     return output

    # self @ diagonal_matrix
    def matmul_diag(self, diagonal):
        col = self.edge_index[1]
        updated_edge_weight = self.edge_weight * tf.gather(diagonal, col)
        return SparseMatrix(self.edge_index, updated_edge_weight, self.shape)

    # self @ diagonal_matrix
    def rmatmul_diag(self, diagonal):
        row = self.edge_index[0]
        updated_edge_weight = tf.gather(diagonal, row) * self.edge_weight
        return SparseMatrix(self.edge_index, updated_edge_weight, self.shape)

    # self @ other (other is a dense tensor or SparseAdj)
    def __matmul__(self, other):
        return self.matmul(other)

    # h @ self (h is a dense tensor)
    def __rmatmul__(self, h):
        return self.rmatmul_dense(h)

    def eliminate_zeros(self):
        edge_index_is_tensor = tf.is_tensor(self.edge_index)
        edge_weight_is_tensor = tf.is_tensor(self.edge_weight)

        mask = tf.not_equal(self.edge_weight, 0.0)
        masked_edge_index = tf.boolean_mask(self.edge_index, mask, axis=1)
        masked_edge_weight = tf.boolean_mask(self.edge_weight, mask)

        if not edge_index_is_tensor:
            masked_edge_index = masked_edge_index.numpy()

        if not edge_weight_is_tensor:
            masked_edge_weight = masked_edge_weight.numpy()

        return SparseMatrix(masked_edge_index, masked_edge_weight, shape=self.shape)

    def merge(self, other_sparse_adj, merge_mode):
        """
        element-wise merge

        :param other_sparse_adj:
        :param merge_mode:
        :return:
        """

        edge_index_is_tensor = tf.is_tensor(self.edge_index)
        edge_weight_is_tensor = tf.is_tensor(self.edge_weight)

        combined_edge_index = tf.concat([self.edge_index, other_sparse_adj.edge_index], axis=1)
        combined_edge_weight = tf.concat([self.edge_weight, other_sparse_adj.edge_weight], axis=0)

        merged_edge_index, [merged_edge_weight] = merge_duplicated_edge(
            combined_edge_index, edge_props=[combined_edge_weight], merge_modes=[merge_mode])

        if not edge_index_is_tensor:
            merged_edge_index = merged_edge_index.numpy()

        if not edge_weight_is_tensor:
            merged_edge_weight = merged_edge_weight.numpy()

        merged_sparse_adj = SparseMatrix(merged_edge_index, merged_edge_weight, shape=self.shape)

        return merged_sparse_adj

    def __add__(self, other_sparse_adj):
        """
        element-wise sparse adj addition: self + other_sparse_adj
        :param other_sparse_adj:
        :return:
        """
        return self.merge(other_sparse_adj, merge_mode="sum")

    def __radd__(self, other_sparse_adj):
        """
        element-wise sparse adj addition: other_sparse_adj + self
        :param other_sparse_adj:
        :return:
        """
        return other_sparse_adj + self

    def __sub__(self, other):
        return self + other.negative()

    def __rsub__(self, other):
        return other - self

    def dropout(self, drop_rate, training=False):
        if training and drop_rate > 0.0:
            edge_weight = tf.compat.v2.nn.dropout(self.edge_weight, drop_rate)
        else:
            edge_weight = self.edge_weight
        return SparseMatrix(self.edge_index, edge_weight=edge_weight, shape=self.shape)

    def softmax(self, axis=-1):

        # reduce by row
        if axis == -1 or axis == 1:
            reduce_index = self.edge_index[0]
            num_reduced = self.shape[0]
        # reduce by col
        elif axis == 0 or axis == -2:
            reduce_index = self.edge_index[1]
            num_reduced = self.shape[1]
        else:
            raise Exception("Invalid axis value: {}, axis shoud be -1, -2, 0, or 1".format(axis))

        normed_edge_weight = segment_softmax(self.edge_weight, reduce_index, num_reduced)

        return SparseMatrix(self.edge_index, normed_edge_weight, shape=self.shape)

    def _to_csr_sparse_matrix(self):

        return sparse_tensor_to_csr_sparse_matrix(
            indices=tf.cast(tf.transpose(self.edge_index, [1, 0]), tf.int64),
            values=self.edge_weight,
            dense_shape=self.shape
        )

    @classmethod
    def from_diagonals(cls, diagonals):
        """
        Construct a SparseAdj from diagonals
        :param diagonals:
        :return:
        """
        num_nodes = tf.shape(diagonals)[0]
        row = tf.range(0, num_nodes, dtype=tf.int32)
        col = row
        edge_index = tf.stack([row, col], axis=0)
        return SparseMatrix(edge_index, diagonals, shape=[num_nodes, num_nodes])

    @classmethod
    def eye(cls, num_nodes):
        """
        Construct a SparseAdj with ones on diagonal
        :param diagonals:
        :return:
        """
        diagonals = tf.ones([num_nodes], dtype=tf.float32)
        return SparseMatrix.from_diagonals(diagonals)

    @classmethod
    def from_sparse_tensor(cls, sparse_tensor: tf.sparse.SparseTensor):
        return SparseMatrix(
            edge_index=tf.transpose(sparse_tensor.indices, [1, 0]),
            edge_weight=sparse_tensor.values,
            shape=sparse_tensor.dense_shape
        )

    def to_sparse_tensor(self):

        sparse_tensor = tf.sparse.SparseTensor(
            indices=tf.cast(tf.transpose(self.edge_index, [1, 0]), tf.int64),
            values=self.edge_weight,
            dense_shape=self.shape
        )
        sparse_tensor = tf.sparse.reorder(sparse_tensor)
        return sparse_tensor

    def to_dense(self):
        return tf.sparse.to_dense(self.to_sparse_tensor())

    def __str__(self):
        return "SparseAdj: \n" \
               "edge_index => \n" \
               "{}\n" \
               "edge_weight => {}\n" \
               "shape => {}".format(self.edge_index, self.edge_weight, self.shape)

    def __repr__(self):
        return self.__str__()
