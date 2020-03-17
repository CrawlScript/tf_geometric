import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# TU Datasets: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
graph_dicts = tfg.datasets.TUDataset("NCI1").load_data()


# Since a TU dataset may contain node_labels, node_attributes etc., each of which can be used as node features
# We process each graph as a dict and return a list of dict for graphs
# You can easily construct you Graph object with the data dict

num_node_labels = np.max([np.max(graph_dict["node_labels"]) for graph_dict in graph_dicts]) + 1


def convert_node_labels_to_one_hot(node_labels):
    num_nodes = len(node_labels)
    x = np.zeros([num_nodes, num_node_labels], dtype=np.float32)
    x[list(range(num_nodes)), node_labels] = 1.0
    return x


def construct_graph(graph_dict):
    return tfg.Graph(
        x=convert_node_labels_to_one_hot(graph_dict["node_labels"]),
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
            yield batch_graph

        if not infinite:
            break


batch_size = 256

drop_rate = 0.2
gin0 = tfg.layers.GIN(64, activation=tf.nn.relu)
gin1 = tfg.layers.GIN(32, activation=tf.nn.relu)
dropout = keras.layers.Dropout(drop_rate)
dense = keras.layers.Dense(num_classes)


def forward(batch_graph, training=False, pooling="mean"):
    # GCN Encoder
    h = gin0([batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight])
    h = dropout(h, training=training)
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

    h = dropout(h, training=training)

    # Predict Graph Labels
    h = dense(h)
    return h


def evaluate():
    preds_list = []
    y_list = []
    for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        logits = forward(test_batch_graph)
        preds = tf.argmax(logits, axis=-1)

        preds_list.append(preds.numpy())
        y_list.append(test_batch_graph.y)

    preds = np.concatenate(preds_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    accuracy = accuracy_score(y, preds)
    return accuracy


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)

for step in range(2000):
    train_batch_graph = next(train_batch_generator)
    with tf.GradientTape() as tape:
        logits = forward(train_batch_graph, training=True)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.one_hot(train_batch_graph.y, depth=num_classes)
        )

    vars = tape.watched_variables()
    grads = tape.gradient(losses, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        mean_loss = tf.reduce_mean(losses)
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, mean_loss, accuracy))