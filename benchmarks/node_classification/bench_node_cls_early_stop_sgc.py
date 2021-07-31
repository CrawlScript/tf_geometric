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

num_classes = graph.y.max() + 1

learning_rate = 0.2
l2_coef = 5e-6
num_steps = 201

if dataset == "citeseer":
    l2_coef = 1e-4
elif dataset == "pubmed":
    l2_coef = 5e-5
    num_steps = 61

model = tfg.layers.SGC(num_classes, k=2)
model.build_cache_for_graph(graph)


# @tf_utils.function can speed up functions for TensorFlow 2.x
@tf_utils.function
def forward(graph, training=False):
    return model([graph.x, graph.edge_index, graph.edge_weight], training=training, cache=graph.cache)


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


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


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

        if val_accuracy > best_val_accuracy and val_loss < min_val_loss:
        # if True:
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

with open("results.txt", "a", encoding="utf-8") as f:
    f.write("{}\n".format(final_test_accuracy))
