# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras
import tf_geometric as tfg
from tf_geometric.datasets import PlanetoidDataset


# Planetoid Datasets: "cora" | "citeseer" | "pubmed"
graph, (train_index, valid_index, test_index) = PlanetoidDataset("cora").load_data()

num_classes = graph.y.max() + 1

gcn0 = tfg.layers.GCN(16, activation=tf.nn.relu)
gcn1 = tfg.layers.GCN(num_classes)

drop_rate = 0.5
dropout = tf.keras.layers.Dropout(drop_rate)

def forward(graph, training=False):
    h = dropout(graph.x, training=training)
    h = gcn0([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
    h = dropout(h, training=training)
    h = gcn1([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
    return h


def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vals = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4


def evaluate():
    logits = forward(graph)
    masked_logits = tf.gather(logits, test_index)
    masked_labels = tf.gather(graph.y, test_index)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(masked_labels, y_pred)
    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)


for step in range(1000):
    with tf.GradientTape() as tape:
        logits = forward(graph, training=True)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))
