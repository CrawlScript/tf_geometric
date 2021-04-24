# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
from tf_geometric.datasets.ppi import PPIDataset
from tf_geometric.utils.graph_utils import RandomNeighborSampler
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

train_graphs, valid_graphs, test_graphs = PPIDataset().load_data()

# traverse all graphs
for graph in train_graphs + valid_graphs + test_graphs:
    neighbor_sampler = RandomNeighborSampler(graph.edge_index)
    graph.cache["sampler"] = neighbor_sampler

num_classes = train_graphs[0].y.shape[1]

graph_sages = [
    # tfg.layers.MaxPoolGraphSage(units=256, activation=tf.nn.relu, concat=True),
    # tfg.layers.MaxPoolGraphSage(units=256, activation=tf.nn.relu, concat=True)

    # tfg.layers.MeanPoolGraphSage(units=256, activation=tf.nn.relu, concat=True),
    # tfg.layers.MeanPoolGraphSage(units=256, activation=tf.nn.relu, concat=True)

    tfg.layers.MeanGraphSage(units=256, activation=tf.nn.relu, concat=True),
    tfg.layers.MeanGraphSage(units=256, activation=tf.nn.relu, concat=True)

    # tfg.layers.SumGraphSage(units=256, activation=tf.nn.relu, concat=True),
    # tfg.layers.SumGraphSage(units=256, activation=tf.nn.relu, concat=True)

    # tfg.layers.LSTMGraphSage(units=256, activation=tf.nn.relu, concat=True),
    # tfg.layers.LSTMGraphSage(units=256, activation=tf.nn.relu, concat=True)

    # tfg.layers.GCNGraphSage(units=256, activation=tf.nn.relu),
    # tfg.layers.GCNGraphSage(units=256, activation=tf.nn.relu)
]

fc = tf.keras.Sequential([
    keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes)
])

num_sampled_neighbors_list = [25, 10]


def forward(graph, training=False):
    neighbor_sampler = graph.cache["sampler"]
    h = graph.x
    for i, (graph_sage, num_sampled_neighbors) in enumerate(zip(graph_sages, num_sampled_neighbors_list)):
        sampled_edge_index, sampled_edge_weight = neighbor_sampler.sample(k=num_sampled_neighbors)
        h = graph_sage([h, sampled_edge_index, sampled_edge_weight], training=training)
    h = fc(h, training=training)
    return h


def compute_loss(logits, vars):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=tf.convert_to_tensor(graph.y, dtype=tf.float32)
    )

    kernel_vars = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vars]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 1e-5


def calc_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0

    return f1_score(y_true, y_pred, average="micro")


def evaluate(graphs):
    y_preds = []
    y_true = []

    for graph in graphs:
        y_true.append(graph.y)
        logits = forward(graph)
        y_preds.append(logits.numpy())

    y_pred = np.concatenate(y_preds, axis=0)
    y = np.concatenate(y_true, axis=0)

    mic = calc_f1(y, y_pred)

    return mic


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

for epoch in tqdm(range(20)):

    for graph in train_graphs:
        with tf.GradientTape() as tape:
            logits = forward(graph, training=True)
            loss = compute_loss(logits, tape.watched_variables())

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

    if epoch % 1 == 0:
        valid_f1_mic = evaluate(valid_graphs)
        test_f1_mic = evaluate(test_graphs)
        print("epoch = {}\tloss = {}\tvalid_f1_micro = {}".format(epoch, loss, valid_f1_mic))
        print("epoch = {}\ttest_f1_micro = {}".format(epoch, test_f1_mic))
# test_f1_mic = evaluate(test_graphs)
# print("test_f1_micro = {}".format(test_f1_mic))
