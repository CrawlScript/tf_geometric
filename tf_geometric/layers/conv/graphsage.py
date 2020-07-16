# coding=utf-8
import tensorflow as tf
from tf_geometric.nn.conv.graphsage import Mean_Aggregator
from tf_geometric.layers.kernel.map_reduce import MapReduceGNN


class GraphSAGE(MapReduceGNN):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
           Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    """

    def __init__(self, hidden_units, aggregate_fun=Mean_Aggregator, activation=tf.nn.relu, use_bias=True,
                 dropout_rate=0.5,
                 normalize=False, *args, **kwargs):
        """

        :param hidden_units: Positive integer, dimensionality of the output space.
        :param aggregate_fun: Aggregator Architectures to use.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param dropout_rate: Dropout rate.
        :param normalize: If set to :obj:`True`, output features
                will be :math:`\ell_2`-normalized, *i.e.*,
                :math:`\frac{\mathbf{x}^{\prime}_i}
                {\| \mathbf{x}^{\prime}_i \|_2}`.
                (default: :obj:`False`)
        """
        super().__init__(*args, **kwargs)
        self.units = hidden_units

        self.aggregate_fun = aggregate_fun
        assert self.aggregate_fun is not None

        self.acvitation = activation
        self.use_bias = use_bias

        self.dropout_rate = dropout_rate
        self.normalize = normalize

    def build(self, input_shape):
        self.aggregate_fun = self.aggregate_fun(units=self.units, activation=self.acvitation, use_bias=self.use_bias,
                                                dropout_rate=self.dropout_rate, normalize=self.normalize)

    def call(self, inputs, cache=None, training=None, mask=None):
        return self.aggregate_fun(inputs, cache=None, training=None, mask=None)
