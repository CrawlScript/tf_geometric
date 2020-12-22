# coding=utf-8
import os

from tf_geometric.utils import tf_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

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

drop_rate = 0.4


class MeanPoolNetwork(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gcn0 = tfg.layers.GCN(64, activation=tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(32, activation=tf.nn.relu)
        self.dropout = keras.layers.Dropout(drop_rate)
        self.dense = keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        x, edge_index, node_graph_index = inputs
        # GCN Encoder
        h = self.gcn0([x, edge_index])
        h = self.dropout(h, training=training)
        h = self.gcn1([h, edge_index])

        # Mean Pooling
        h = tfg.nn.mean_pool(h, node_graph_index)
        h = self.dropout(h, training=training)

        # Predict Graph Labels
        h = self.dense(h)
        return h


model = MeanPoolNetwork()

def forward(batch_graph, training=False):
    return model([batch_graph.x, batch_graph.edge_index, batch_graph.node_graph_index], training=training)

def evaluate():
    accuracy_m = keras.metrics.Accuracy()

    for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        logits = forward(test_batch_graph)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(test_batch_graph.y, preds)

    return accuracy_m.result().numpy()


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
