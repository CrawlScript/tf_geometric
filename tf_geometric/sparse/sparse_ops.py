# coding=utf-8

import tensorflow as tf
from tf_geometric.sparse.sparse_matrix import SparseMatrix


# sparse_adj @ diagonal_matrix
def sparse_diag_matmul(sparse: SparseMatrix, diagonal):
    return sparse.matmul_diag(diagonal)


# self @ diagonal_matrix
def diag_sparse_matmul(diagonal, sparse: SparseMatrix):
    return sparse.rmatmul_diag(diagonal)


# element-wise sparse_adj addition
def add(a: SparseMatrix, b: SparseMatrix):
    return a + b


# element-wise sparse_adj subtraction
def subtract(a: SparseMatrix, b: SparseMatrix):
    return a - b


# element-wise maximum(a, b)
def maximum(a: SparseMatrix, b: SparseMatrix):
    return a.merge(b, merge_mode="max")


# element-wise minimum(a, b)
def minimum(a: SparseMatrix, b: SparseMatrix):
    return a.merge(b, merge_mode="min")


# Construct a SparseAdj from diagonals
def diags(diagonals):
    return SparseMatrix.from_diagonals(diagonals)


# Construct a SparseAdj with ones on diagonal
def eye(num_nodes):
    return SparseMatrix.eye(num_nodes)