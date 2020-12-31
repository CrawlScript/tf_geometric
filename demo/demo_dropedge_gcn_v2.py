# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_geometric.layers import GCN, DropEdge
from tensorflow.keras.layers import Dropout
from tf_geometric.utils import tf_utils
import tensorflow as tf
import tf_geometric as tfg
from tqdm import tqdm
import time

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()

num_classes = graph.y.max() + 1
num_hidden_layer = 1
num_base_layer = 8
units = 128
drop_rate = 0.8
edge_drop_rate = 0.5
learning_rate = 1e-2
l2_coe = 0.0


class GCN_BS(tf.keras.Model):
    """
    GCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(self, units, activation=None, use_bn=True, use_loop=True, use_bias=True, res=False, *args, **kwargs):
        """
         Initial function.
        :param units: the output feature dimension.
        :param activation: the activation function.
        :param use_bn: using batch normalization.
        :param use_loop: using self feature modeling.
        :param use_bias: enable bias.
        :param res: enable res connections.
        """
        super().__init__(*args, **kwargs)
        self.units = units
        self.gcn = GCN(self.units, renorm=False, use_bias=False)
        self.activation = activation
        self.use_bn = use_bn
        self.use_loop = use_loop
        self.use_bias = use_bias
        self.res = res
        if self.use_loop:
            self.self_weight = tf.keras.layers.Dense(self.units, use_bias=False)
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization()
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

    def call(self, inputs, training=None, mask=None, cache=None):
        x = inputs[0]
        h = self.gcn(inputs, cache=cache)

        if self.use_loop:
            h += self.self_weight(x)
        if self.use_bias:
            h += self.bias
        if self.use_bn:
            h = self.bn(h, training=training)
        if self.activation:
            h = self.activation(h)
        if self.res:
            h += x
        return h


class GCNBaseBlock(tf.keras.Model):
    """
    The base block for Multi-layer GCN
    """
    def __init__(self, units, nbaselayer,
                 use_bn=True, use_loop=True, activation=tf.nn.relu, drop_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = Dropout(drop_rate)
        self.gcns = [GCN_BS(units, activation, use_bn, use_loop) for _ in range(nbaselayer)]

    def call(self, inputs, training=None, mask=None, cache=None):
        h, *edge_attrs = inputs

        outputs = []
        for i in range(len(self.gcns)):
            h = self.gcns[i]([h, *edge_attrs], cache=cache, training=training)
            h = self.dropout(h, training=training)
            outputs.append(h)

        h = tf.concat(outputs, axis=-1)
        return h


# Multi-layer DropEdge GCN Model
class DropEdgeGCNModel(tf.keras.Model):

    def __init__(self, units, activation, nhidlayer, nbaselayer, drop_rate=0.5, edge_drop_rate=0.5, use_bn=True, use_loop=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.units = units
        self.activation = activation
        self.use_bn = use_bn
        self.use_loop = use_loop
        self.nhidlayer = nhidlayer
        self.nbaselayer = nbaselayer
        self.drop_rate = drop_rate
        self.edge_drop_rate = edge_drop_rate

        self.in_gcn = GCN_BS(units=self.units, activation=self.activation, use_bn=self.use_bn, use_loop=self.use_loop)
        self.out_gcn = GCN_BS(units=num_classes, use_bn=self.use_bn, use_loop=self.use_loop)
        self.hidden_gcns = [
            GCNBaseBlock(units=units, nbaselayer=self.nbaselayer, activation=self.activation, drop_rate=self.drop_rate,
                         use_bn=self.use_bn, use_loop=self.use_loop) for _ in range(self.nhidlayer)]

        self.dropout = Dropout(self.drop_rate)
        self.dropedge = DropEdge(self.edge_drop_rate, force_undirected=False)

    def call(self, inputs, training=None, mask=None):
        h, edge_index, edge_weight = inputs

        # DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
        edge_index, edge_weight = self.dropedge([edge_index, edge_weight], training=training)
        cache = {}
        h = self.in_gcn([h, edge_index, edge_weight], cache=cache, training=training)
        h = self.dropout(h, training=training)

        cache = {}
        for i in range(num_hidden_layer):
            h = self.hidden_gcns[i]([h, edge_index, edge_weight], cache=cache, training=training)

        h = self.out_gcn([h, edge_index, edge_weight], cache=cache, training=training)
        return h


model = DropEdgeGCNModel(
    units=units,
    activation=tf.nn.relu,
    nhidlayer=num_hidden_layer,
    nbaselayer=num_base_layer,
    drop_rate=drop_rate,
    edge_drop_rate=edge_drop_rate,
    use_bn=True,
    use_loop=True
)


# @tf_utils.function can speed up functions for TensorFlow 2.x.
# @tf_utils.function is not compatible with TensorFlow 1.x and dynamic graph.cache.
@tf_utils.function
def forward(graph, training=False):
    return model([graph.x, graph.edge_index, graph.edge_weight], training=training)


@tf.function
def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vals = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * l2_coe


@tf.function
def evaluate():
    logits = forward(graph)
    masked_logits = tf.gather(logits, test_index)
    masked_labels = tf.gather(graph.y, test_index)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    corrects = tf.equal(y_pred, masked_labels)
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for step in range(1, 1501):
    with tf.GradientTape() as tape:
        logits = forward(graph, training=True)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

print("\nstart speed test...")
num_test_iterations = 1000
start_time = time.time()
for _ in tqdm(range(num_test_iterations)):
    logits = forward(graph)
end_time = time.time()
print("mean forward time: {} seconds".format((end_time - start_time) / num_test_iterations))

if tf.__version__[0] == "1":
    print("** @tf_utils.function is disabled in TensorFlow 1.x. "
          "Upgrade to TensorFlow 2.x for 10X faster speed. **")
