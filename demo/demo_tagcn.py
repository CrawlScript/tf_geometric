# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_geometric.utils import tf_utils
import tf_geometric as tfg
import tensorflow as tf
import numpy as np
from tf_geometric.datasets import CoraDataset

graph, (train_index, valid_index, test_index) = CoraDataset().load_data()

num_classes = graph.y.max() + 1
drop_rate = 0.3


# Multi-layer GCN Model
class TAGCNModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tagcn0 = tfg.layers.TAGCN(16, activation=tf.nn.relu)
        self.tagcn1 = tfg.layers.TAGCN(num_classes)
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, edge_weight = inputs
        h = self.tagcn0([x, edge_index, edge_weight], cache=graph.cache)
        h = self.dropout(h, training=training)
        h = self.tagcn1([h, edge_index, edge_weight], cache=graph.cache)
        return h


model = TAGCNModel()


# @tf_utils.function can speed up functions for TensorFlow 2.x.
# @tf_utils.function is not compatible with TensorFlow 1.x and dynamic graph.cache.
@tf_utils.function
def forward(graph, training=False):
    return model([graph.x, graph.edge_index, graph.edge_weight], training=training, cache=graph.cache)


# The following line is only necessary for using GCN with @tf_utils.function
# For usage without @tf_utils.function, you can commont the following line and GCN layers can automatically manager the cache
model.tagcn0.cache_normed_edge(graph)


@tf_utils.function
def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vars = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vars]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4


@tf_utils.function
def evaluate(mask):
    logits = forward(graph)
    masked_logits = tf.gather(logits, mask)
    masked_labels = tf.gather(graph.y, mask)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    corrects = tf.equal(y_pred, masked_labels)
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

best_test_acc = tmp_valid_acc = 0
for step in range(1, 101):
    with tf.GradientTape() as tape:
        logits = forward(graph, training=True)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    valid_acc = evaluate(valid_index)
    test_acc = evaluate(test_index)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        tmp_valid_acc = valid_acc
    print("step = {}\tloss = {}\tvalid_acc = {}\tbest_test_acc = {}".format(step, loss, tmp_valid_acc, best_test_acc))
