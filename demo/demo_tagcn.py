# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras
from tf_geometric.layers.conv.tagcn import TAGCN
from tf_geometric.datasets.cora import CoraDataset


graph, (train_index, valid_index, test_index) = CoraDataset().load_data()

num_classes = graph.y.max() + 1

tagcn0 = TAGCN(16, K=2, activation=tf.nn.relu)
tagcn1 = TAGCN(num_classes, K=2)
dropout = keras.layers.Dropout(0.5)

def forward(graph, training=False):
    h = tagcn0([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache)
    h = dropout(h, training=training)
    h = tagcn1([h, graph.edge_index, graph.edge_weight], cache=graph.cache)
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


def evaluate(mask):
    logits = forward(graph)
    logits = tf.nn.log_softmax(logits, axis=-1)
    masked_logits = tf.gather(logits, mask)
    masked_labels = tf.gather(graph.y, mask)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(masked_labels, y_pred)
    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

best_test_acc = tmp_valid_acc = 0
for step in range(1,101):
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