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
num_gcns = 8
drop_rate = 0.5
edge_drop_rate = 0.8
learning_rate = 5e-3
l2_coe = 0.0

units_list = [256] * (num_gcns - 1) + [num_classes]


# Multi-layer DropEdge GCN Model
class DropEdgeGCNModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        activations = [tf.nn.relu if i < len(units_list) - 1 else None for i in range(len(units_list))]

        self.gcns = [GCN(units, activation=activation) for units, activation in zip(units_list, activations)]

        self.dropout = Dropout(drop_rate)
        self.dropedge = DropEdge(edge_drop_rate, force_undirected=False)

    def call(self, inputs, training=None, mask=None):
        h, edge_index, edge_weight = inputs

        # DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
        edge_index, edge_weight = self.dropedge([edge_index, edge_weight], training=training)
        h = self.dropout(h, training=training)

        cache = {}
        for i in range(num_gcns):
            h = self.gcns[i]([h, edge_index, edge_weight], cache=cache)

        return h


model = DropEdgeGCNModel()


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

for step in range(1, 1001):
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
