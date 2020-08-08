from scipy.sparse.linalg import eigs, eigsh
from tf_geometric.utils.graph_utils import get_laplacian, to_scipy_sparse_matrix, remove_self_loop_edge, add_self_loop_edge


class LaplacianLambdaMax(object):
    r"""Computes the highest eigenvalue of the graph Laplacian given by
    :meth:`torch_geometric.utils.get_laplacian`.

    Args:
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of the largest eigenvalue. (default: :obj:`False`)
    """

    def __init__(self, normalization_type=None, is_undirected=False):
        assert normalization_type in [None, 'sym', 'rw'], 'Invalid normalization'
        self.normalization = normalization_type
        self.is_undirected = is_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_weight = data.edge_weight

        edge_index, edge_weight = remove_self_loop_edge(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight,
                                                self.normalization,
                                                num_nodes=data.x.shape[0])
        # edge_index, edge_weight = add_self_loop_edge(edge_index, data.x.shape[0], edge_weight, fill_weight=-1.)

        L = to_scipy_sparse_matrix(edge_index, edge_weight, data.x.shape[0])

        eig_fn = eigs
        if self.is_undirected and self.normalization != 'rw':
            eig_fn = eigsh

        lambda_max = eig_fn(L, k=1, which='LM', return_eigenvectors=False)
        data.lambda_max = float(lambda_max)

        return data

    def __repr__(self):
        return '{}(normalization={})'.format(self.__class__.__name__,
                                             self.normalization)
