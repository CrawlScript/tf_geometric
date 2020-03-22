# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()

num_classes = graph.y.max() + 1
drop_rate = 0.2

gat0 = tfg.layers.GAT(64, activation=tf.nn.relu, num_heads=8, drop_rate=drop_rate, attention_units=8)
gat1 = tfg.layers.GAT(num_classes, drop_rate=0.6, attention_units=1)
dropout = keras.layers.Dropout(drop_rate)


def forward(graph, training=False):
    h = graph.x
    h = dropout(h, training=training)
    h = gat0([h, graph.edge_index], training=training)
    h = dropout(h, training=training)
    h = gat1([h, graph.edge_index], training=training)
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


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

for step in range(2000):
    with tf.GradientTape() as tape:
        logits = forward(graph, training=True)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))