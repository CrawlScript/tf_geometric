import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# TU Datasets: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
# COLLAB is a large dataset, which may costs 5 minutes for processing.
# tfg will automatically cache the processing result after the first processing.
# Thus, you can load it with only few seconds then.
graph_dicts = tfg.datasets.TUDataset("COLLAB").load_data()


# Since a TU dataset may contain node_labels, node_attributes etc., each of which can be used as node features
# We process each graph as a dict and return a list of dict for graphs
# You can easily construct you Graph object with the data dict


def create_fake_node_features(num_nodes):
    x = np.ones([num_nodes, 1], dtype=np.float32)
    return x


def construct_graph(graph_dict):
    return tfg.Graph(
        x=create_fake_node_features(graph_dict["num_nodes"]),
        edge_index=graph_dict["edge_index"],
        y=graph_dict["graph_label"]  # graph_dict["graph_label"] is a list with one int element
    )


graphs = [construct_graph(graph_dict) for graph_dict in graph_dicts]
num_classes = np.max([graph.y[0] for graph in graphs]) + 1

train_graphs, test_graphs = train_test_split(graphs, test_size=0.1)


def create_graph_generator(graphs, batch_size, infinite=False, shuffle=False):
    while True:
        dataset = tf.data.Dataset.range(len(graphs))
        if shuffle:
            dataset = dataset.shuffle(2000)
        dataset = dataset.batch(batch_size)

        for batch_graph_index in dataset:
            batch_graph_list = [graphs[i] for i in batch_graph_index]

            batch_graph = tfg.BatchGraph.from_graphs(batch_graph_list)
            # print("num_nodes: ", batch_graph.num_nodes)
            yield batch_graph

        if not infinite:
            break


batch_size = 100

drop_rate = 0.2
gin0 = tfg.layers.GIN(100, activation=tf.nn.relu)
gin1 = tfg.layers.GIN(100, activation=tf.nn.relu)
mlp = keras.Sequential([
    keras.layers.Dense(50),
    keras.layers.Dropout(drop_rate),
    keras.layers.Dense(num_classes)
])
# dense = keras.layers.Dense(num_classes)


def forward(batch_graph, training=False, pooling="sum"):
    # GCN Encoder
    h = gin0([batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight])
    h = gin1([h, batch_graph.edge_index, batch_graph.edge_weight])

    # Pooling
    if pooling == "mean":
        h = tfg.nn.mean_pool(h, batch_graph.node_graph_index)
    elif pooling == "sum":
        h = tfg.nn.mean_pool(h, batch_graph.node_graph_index)
    elif pooling == "max":
        h = tfg.nn.max_pool(h, batch_graph.node_graph_index)
    elif pooling == "min":
        h = tfg.nn.min_pool(h, batch_graph.node_graph_index)

    # Predict Graph Labels
    h = mlp(h, training=training)
    return h


def evaluate():
    accuracy_m = keras.metrics.Accuracy()

    for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        logits = forward(test_batch_graph)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(test_batch_graph.y, preds)

    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)

import time
for step in range(20000):
    train_batch_graph = next(train_batch_generator)
    with tf.GradientTape() as tape:
        logits = forward(train_batch_graph, training=True)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.one_hot(train_batch_graph.y, depth=num_classes)
        )

        kernel_vals = [var for var in tape.watched_variables() if "kernel" in var.name]
        l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]

        loss = tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))