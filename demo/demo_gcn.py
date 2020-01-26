# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras
from tf_geometric.datasets.cora import CoraDataset
from tf_geometric.layers import GCN

graph, (train_index, valid_index, test_index) = CoraDataset().load_data()

num_classes = graph.y.max() + 1

gcn0 = GCN(32, activation=tf.nn.relu)
gcn1 = GCN(num_classes)


def forward(graph):
    h = gcn0([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)
    h = gcn1([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
    return h


def compute_losses(logits, mask_index):
    masked_logits = tf.gather(logits, mask_index)
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )
    return losses


def evaluate():
    logits = forward(graph)
    masked_logits = tf.gather(logits, test_index)
    masked_labels = tf.gather(graph.y, test_index)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    corrects = tf.cast(tf.equal(y_pred, masked_labels), tf.float32)
    accuracy = tf.reduce_mean(corrects)
    return accuracy


optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)

for step in range(1000):
    with tf.GradientTape() as tape:
        logits = forward(graph)
        losses = compute_losses(logits, train_index)

    vars = tape.watched_variables()
    grads = tape.gradient(losses, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        mean_loss = tf.reduce_mean(losses)
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, mean_loss, accuracy))