# coding=utf-8

import tensorflow as tf
from tf_geometric.sparse.sparse_adj import SparseAdj


# sparse_adj @ diagonal_matrix
def sparse_diag_matmul(sparse_adj: SparseAdj, diagonal):
    return sparse_adj.matmul_diag(diagonal)


# self @ diagonal_matrix
def diag_sparse_matmul(diagonal, sparse_adj: SparseAdj):
    return sparse_adj.rmatmul_diag(diagonal)


