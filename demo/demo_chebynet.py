# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_geometric.utils import tf_utils
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from tf_geometric.datasets import CoraDataset
from tqdm import tqdm

graph, (train_index, valid_index, test_index) = CoraDataset().load_data()

num_classes = graph.y.max() + 1

model = tfg.layers.ChebyNet(64, k=3, activation=tf.nn.relu)
fc = tf.keras.Sequential([
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes)
])

model.cache_normed_edge(graph)


# @tf_utils.function can speed up functions for TensorFlow 2.x
@tf_utils.function
def forward(graph, training=False):
    h = model([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)
    h = fc(h, training=training)
    return h


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


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)


@tf_utils.function
def train_step():
    with tf.GradientTape() as tape:
        logits = forward(graph, training=True)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return loss


@tf_utils.function
def evaluate():
    logits = forward(graph)
    masked_logits = tf.gather(logits, test_index)
    masked_labels = tf.gather(graph.y, test_index)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    corrects = tf.equal(y_pred, masked_labels)
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy


best_test_acc = 0
for step in tqdm(range(1, 101)):
    loss = train_step()

    test_acc = evaluate()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
    print("step = {}\tloss = {}\tbest_test_acc = {}".format(step, loss, best_test_acc))
