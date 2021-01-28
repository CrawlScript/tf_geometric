# coding=utf-8
import os

from tf_geometric.layers import SAGPool, GCN, SortPool, MeanGraphSage
from tf_geometric.utils import tf_utils
from tf_geometric.utils.graph_utils import convert_x_to_3d

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tf_geometric as tfg
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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


batch_size = 512


# SAGPool_h
class SortPoolModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gcns = [GCN(128, activation=tf.nn.relu) for _ in range(3)]

        self.K = 30
        self.sort_pool = SortPool(K=self.K)
        self.conv1d = tf.keras.layers.Conv1D(128, 5, activation=tf.nn.relu)
        self.flatten = tf.keras.layers.Flatten()

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, inputs, training=None, mask=None):

        x, edge_index, edge_weight, node_graph_index = inputs
        h = x
        for gcn in self.gcns:
            h = gcn([h, edge_index, edge_weight], training=training)

        pooled_x, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index = \
            self.sort_pool([h, edge_index, edge_weight, node_graph_index], training=training)
        h = convert_x_to_3d(pooled_x, node_graph_index, K=self.K)
        h = self.conv1d(h)
        # h = tf.reduce_max(h, axis=1)
        h = self.flatten(h)

        # Predict Graph Labels
        h = self.mlp(h, training=training)
        return h


model = SortPoolModel()


def forward(batch_graph, training=False):
    return model([batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight, batch_graph.node_graph_index],
                 training=training)


def evaluate():
    accuracy_m = tf.keras.metrics.Accuracy()

    for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        logits = forward(test_batch_graph)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(test_batch_graph.y, preds)

    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)


for step in tqdm(range(20000)):
    train_batch_graph = next(train_batch_generator)
    with tf.GradientTape() as tape:
        logits = forward(train_batch_graph, training=True)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.one_hot(train_batch_graph.y, depth=num_classes)
        )
        mean_loss = tf.reduce_mean(losses)

    vars = tape.watched_variables()
    grads = tape.gradient(mean_loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, mean_loss, accuracy))

