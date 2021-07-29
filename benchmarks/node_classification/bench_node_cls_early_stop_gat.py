# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_geometric.utils import tf_utils
import tf_geometric as tfg
import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np

dataset = "cora"
# dataset = "citeseer"
# dataset = "pubmed"

graph, (train_index, valid_index, test_index) = tfg.datasets.PlanetoidDataset(dataset).load_data()

num_steps = 401

num_classes = graph.y.max() + 1
# att_drop_rate = 0.6
drop_rate = 0.6
if dataset == "citeseer":
    drop_rate = 0.6
    l2_coef = 2e-3
elif dataset == "cora":
    drop_rate = 0.7
    l2_coef = 1e-3
elif dataset == "pubmed":
    drop_rate = 0.0
    l2_coef = 2e-3
    # num_steps = 1001

patience = 20


# Multi-layer GAT Model
class GATModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if dataset != "pubmed":
            self.gat0 = tfg.layers.GAT(64, activation=tf.nn.relu, num_heads=8, drop_rate=drop_rate, attention_units=8)
            self.gat1 = tfg.layers.GAT(num_classes, drop_rate=drop_rate, attention_units=1)
        else:
            self.gat0 = tfg.layers.GAT(64, activation=tf.nn.relu, num_heads=1, drop_rate=drop_rate, attention_units=1)
            self.gat1 = tfg.layers.GAT(num_classes, drop_rate=drop_rate, num_heads=8, attention_units=8,
                                       split_value_heads=False)

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs
        h = self.dropout(x, training=training)
        h = self.gat0([h, edge_index], training=training)
        h = self.dropout(h, training=training)
        h = self.gat1([h, edge_index], training=training)
        return h


model = GATModel()


# @tf_utils.function can speed up functions for TensorFlow 2.x
@tf_utils.function
def forward(graph, training=False):
    return model([graph.x, graph.edge_index], training=training)


@tf_utils.function
def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vals = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
    cls_loss = tf.reduce_mean(losses)
    l2_loss = tf.add_n(l2_losses)
    return cls_loss + l2_loss * l2_coef, cls_loss, l2_loss


@tf_utils.function
def evaluate(current_test_index):
    with tf.GradientTape() as tape:
        logits = forward(graph)
    loss = compute_loss(logits, current_test_index, tape.watched_variables())
    masked_logits = tf.gather(logits, current_test_index)
    masked_labels = tf.gather(graph.y, current_test_index)
    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    corrects = tf.equal(y_pred, masked_labels)
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy, loss


@tf_utils.function
def evaluate_test():
    return evaluate(test_index)


@tf_utils.function
def evaluate_val():
    return evaluate(valid_index)


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)


@tf_utils.function
def train_step():
    with tf.GradientTape() as tape:
        logits = forward(graph, training=True)
        loss, _, _ = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return loss


val_accuracy_list = []
test_accuracy_list = []
loss_list = []

best_val_accuracy = 0
min_val_loss = 1000

final_test_accuracy = None
final_step = None

patience_counter = 0

for step in range(1, num_steps):

    loss = train_step()

    if step % 1 == 0:
        test_accuracy, _ = evaluate_test()
        val_accuracy, (_, val_loss, _) = evaluate_val()

        val_accuracy = val_accuracy.numpy()
        val_loss = val_loss.numpy()

        # if val_accuracy > best_val_accuracy and val_loss < min_val_loss:
        if val_accuracy > best_val_accuracy and val_loss < min_val_loss:
            final_test_accuracy = test_accuracy
            final_step = step

            best_val_accuracy = val_accuracy
            min_val_loss = val_loss

        val_accuracy_list.append(val_accuracy)
        test_accuracy_list.append(test_accuracy)
        loss_list.append(val_loss)

        print(
            "step = {}\tloss = {:.4f}\tval_accuracy = {:.4f}\tval_loss = {:.4f}\t"
            "test_accuracy = {:.4f}\tfinal_test_accuracy = {:.4f}\tfinal_step = {}"
            .format(step, loss, val_accuracy, val_loss, test_accuracy, final_test_accuracy, final_step))

print("final accuracy: {}\tfinal_step: {}".format(final_test_accuracy, final_step))
