# coding=utf-8
import tensorflow as tf
# from tf_geometric.sparse.sparse_adj import SparseAdj
from tf_sparse import SparseMatrix

from tf_geometric.nn.conv.gcn import gcn_norm_adj
import tf_sparse as tfs


def tagcn(x, edge_index, edge_weight, k, kernel, bias=None, activation=None, renorm=False, improved=False, cache=None):
    """
    Functional API for Topology Adaptive Graph Convolutional Network (TAGCN).

    :param x: Tensor, shape: [num_nodes, num_features], node features.
    :param edge_index: Tensor, shape: [2, num_edges], edge information.
    :param edge_weight: Tensor or None, shape: [num_edges].
    :param k: Number of hops.(default: :obj:`3`)
    :param kernel: Tensor, shape: [num_features, num_output_features], weight.
    :param bias: Tensor, shape: [num_output_features], bias.
    :param activation: Activation function to use.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """

    num_nodes = tfs.shape(x)[0]

    sparse_adj = SparseMatrix(edge_index, edge_weight, [num_nodes, num_nodes])
    normed_sparse_adj = gcn_norm_adj(sparse_adj, renorm=renorm, improved=improved, cache=cache)

    if isinstance(x, tf.sparse.SparseTensor):
        x = tf.sparse.to_dense(x)
    elif isinstance(x, tfs.SparseMatrix):
        x = x.to_dense()

    xs = [x]
    for _ in range(k):
        h = normed_sparse_adj @ xs[-1]
        xs.append(h)

    h = tf.concat(xs, axis=-1)

    out = h @ kernel
    if bias is not None:
        out += bias

    if activation is not None:
        out = activation(out)

    return out
