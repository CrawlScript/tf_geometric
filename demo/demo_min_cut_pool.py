# coding=utf-8
import os

from tf_geometric.layers import MinCutPool

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
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


batch_size = 50


# Multi-layer GCN Model
class GCNModel(tf.keras.Model):

    def __init__(self, units_list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gcns = [
            # tfg.layers.GCN(units, activation=tf.nn.relu if i < len(units_list) - 1 else None)
            # tfg.layers.GCN(units, activation=tf.nn.relu)
            tfg.layers.SumGraphSage(units, concat=False, activation=tf.nn.relu)
            for i, units in enumerate(units_list)
        ]

    def call(self, inputs, training=None, mask=None):
        x, edge_index, edge_weight = inputs
        h = x
        for gcn in self.gcns:
            h = gcn([h, edge_index, edge_weight], training=training)
        return h


class AssignGCNModel(tf.keras.Model):

    def __init__(self, gcn_units_list, mlp_units_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn_model = GCNModel(units_list=gcn_units_list)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation=tf.nn.relu if i < len(mlp_units_list) - 1 else None)
            for i, units in enumerate(mlp_units_list)
        ])
    def call(self, inputs, training=None, mask=None):
        x, edge_index, edge_weight = inputs
        h = self.gcn_model([x, edge_index, edge_weight], training=training)
        h = self.mlp(h, training=training)
        return h


class MinCutPoolModel(tf.keras.Model):

    def __init__(self, num_clusters_list, num_features_list, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.min_cut_pools = []

        for i, (num_features, num_clusters) in enumerate(zip(num_features_list, num_clusters_list)):
            assign_gcn_model = AssignGCNModel([num_features, num_features], [num_clusters])
            min_cut_pool = MinCutPool(
                feature_gnn=assign_gcn_model.gcn_model,
                assign_gnn=assign_gcn_model, activation=tf.nn.relu,
                units=num_features, num_clusters=num_clusters
            )
            self.min_cut_pools.append(min_cut_pool)

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, inputs, training=None, mask=None, cache=None, return_losses=False):
        x, edge_index, edge_weight, node_graph_index = inputs
        h = x
        graph_h_list = []
        mean_cut_loss_list = []
        mean_orth_loss_list = []
        for min_cut_pool in self.min_cut_pools:
            (pooled_h, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index), loss_func = \
                min_cut_pool([h, edge_index, edge_weight, node_graph_index], training=training, cache=cache,
                             return_loss_func=True)

            graph_h = tf.concat([
                tfg.nn.mean_pool(pooled_h, pooled_node_graph_index),
                tfg.nn.max_pool(pooled_h, pooled_node_graph_index)
            ], axis=-1)
            graph_h_list.append(graph_h)

            if return_losses:
                min_cut_loss, orth_loss = loss_func()
                mean_cut_loss_list.append(min_cut_loss)
                mean_orth_loss_list.append(orth_loss)

            h, edge_index, edge_weight, node_graph_index = pooled_h, pooled_edge_index, pooled_edge_weight, pooled_node_graph_index

        graph_h = tf.concat(graph_h_list, axis=-1)
        logits = self.mlp(graph_h, training=training)

        if not return_losses:
            return logits
        else:
            cut_loss = tf.add_n(mean_cut_loss_list)
            orth_loss = tf.add_n(mean_orth_loss_list)
            return logits, (cut_loss, orth_loss)


num_clusters_list = [20, 5]
num_features_list = [128, 128]

model = MinCutPoolModel(num_clusters_list, num_features_list, num_classes)


def forward(batch_graph, training=False, return_losses=False):
    return model([batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight, batch_graph.node_graph_index],
                 training=training, return_losses=return_losses)


def evaluate():
    accuracy_m = keras.metrics.Accuracy()

    for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
        logits = forward(test_batch_graph)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(test_batch_graph.y, preds)

    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)

for step in tqdm(range(20000)):
    train_batch_graph = next(train_batch_generator)
    with tf.GradientTape() as tape:
        logits, (cut_loss, orth_loss) = forward(train_batch_graph, training=True, return_losses=True)
        cls_losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.one_hot(train_batch_graph.y, depth=num_classes)
        )
        cls_loss = tf.reduce_mean(cls_losses)
        loss = cls_loss + cut_loss + orth_loss

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        accuracy = evaluate()
        print(
            "step = {}\tloss = {}\tcut_loss = {}\torth_loss = {}\taccuracy = {}".format(step, loss, cut_loss, orth_loss,
                                                                                        accuracy))
