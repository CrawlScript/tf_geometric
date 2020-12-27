import os

from tf_geometric.utils import tf_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# TU Datasets: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
# COLLAB is a large dataset, which may costs 5 minutes for processing.
# tfg will automatically cache the processing result after the first processing.
# Thus, you can load it with only few seconds then.
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


class GINPoolNetwork(keras.Model):
    def __init__(self, num_gins, units, num_classes, *args, **kwargs):
        """
        Demo GIN based Pooling Model
        :param num_gins: number of GIN layers
        :param units: Positive integer, dimensionality of the each GIN layer.
        :param num_classes: number of classes (for graph classification)
        """
        super().__init__(*args, **kwargs)

        self.gins = [
            tfg.layers.GIN(
                keras.Sequential([
                    keras.layers.Dense(units, activation=tf.nn.relu),
                    keras.layers.Dense(units),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation(tf.nn.relu)
                ])
            )
            for _ in range(num_gins)  # num_gins blocks
        ]

        self.mlp = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes)
        ])

    # @tf_utils.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, mask=None):

        if len(inputs) == 4:
            x, edge_index, edge_weight, node_graph_index = inputs
        else:
            x, edge_index, node_graph_index = inputs
            edge_weight = None

        hidden_outputs = []
        h = x

        for gin in self.gins:
            h = gin([h, edge_index, edge_weight], training=training)
            hidden_outputs.append(h)

        h = tf.concat(hidden_outputs, axis=-1)
        h = tfg.nn.sum_pool(h, node_graph_index)
        logits = self.mlp(h, training=training)
        return logits


model = GINPoolNetwork(5, 32, num_classes)
batch_size = len(train_graphs)


def evaluate(graphs, batch_size):
    accuracy_m = keras.metrics.Accuracy()

    for batch_graph in create_graph_generator(graphs, batch_size, shuffle=False, infinite=False):
        inputs = [batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight,
                  batch_graph.node_graph_index]
        logits = model(inputs)
        preds = tf.argmax(logits, axis=-1)
        accuracy_m.update_state(batch_graph.y, preds)

    return accuracy_m.result().numpy()


# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)
train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)


best_test_acc = 0
for step in tqdm(range(0, 1000)):
    batch_graph = next(train_batch_generator)
    with tf.GradientTape() as tape:
        inputs = [batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight,
                  batch_graph.node_graph_index]
        logits = model(inputs, training=True)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.one_hot(batch_graph.y, depth=num_classes)
        )

        loss = tf.reduce_mean(losses)
    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 10 == 0:
        train_acc = evaluate(train_graphs, batch_size)
        test_acc = evaluate(test_graphs, batch_size)

        if best_test_acc < test_acc:
            best_test_acc = test_acc

        print("step = {}\tloss = {}\ttrain_acc = {}\ttest_acc={}".format(step, loss, train_acc, best_test_acc))
