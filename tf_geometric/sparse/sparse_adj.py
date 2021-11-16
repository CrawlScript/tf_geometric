# coding=utf-8


"""
Sparse Adj for Computation
"""
from tf_sparse import SparseMatrix

from tf_geometric.nn.kernel.segment import segment_softmax
from tf_geometric.utils.graph_utils import add_self_loop_edge, remove_self_loop_edge


class SparseAdj(SparseMatrix):

    def __init__(self, index, value=None, shape=None, merge=False,
                 edge_weight=None):
        if value is not None and edge_weight is not None:
            raise Exception("\"edge_weight\" is an alias for \"value\", and it is deprecated. You should only provide \"value\".")
        super().__init__(index, value, shape, merge)

    @property
    def edge_index(self):
        return self.index

    @property
    def edge_weight(self):
        return self.value

    def add_self_loop(self, fill_weight=1.0):
        num_nodes = self._shape[0]
        updated_edge_index, updated_edge_weight = add_self_loop_edge(self.index, num_nodes,
                                                                     edge_weight=self.value,
                                                                     fill_weight=fill_weight)
        return self.__class__(updated_edge_index, updated_edge_weight, self._shape)

    def remove_self_loop(self):
        updated_edge_index, updated_edge_weight = remove_self_loop_edge(self.index, edge_weight=self.value)
        return self.__class__(updated_edge_index, updated_edge_weight, self._shape)
