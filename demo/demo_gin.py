import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(2020)
tf.random.set_seed(2020)
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


class GIN(keras.Model):
    def __init__(self, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn1 = keras.Sequential([keras.layers.Dense(hidden_dim, activation=tf.nn.relu), keras.layers.Dense(hidden_dim)])
        self.gin0 = tfg.layers.GIN(nn1)
        nn2 = keras.Sequential([keras.layers.Dense(hidden_dim, activation=tf.nn.relu), keras.layers.Dense(hidden_dim)])
        self.gin1 = tfg.layers.GIN(nn2)
        nn3 = keras.Sequential([keras.layers.Dense(hidden_dim, activation=tf.nn.relu), keras.layers.Dense(hidden_dim)])
        self.gin2 = tfg.layers.GIN(nn3)
        nn4 = keras.Sequential([keras.layers.Dense(hidden_dim, activation=tf.nn.relu), keras.layers.Dense(hidden_dim)])
        self.gin3 = tfg.layers.GIN(nn4)
        nn5 = keras.Sequential([keras.layers.Dense(hidden_dim, activation=tf.nn.relu), keras.layers.Dense(hidden_dim)])
        self.gin4 = tfg.layers.GIN(nn5)

        self.bn0 = keras.layers.BatchNormalization()
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()
        self.bn4 = keras.layers.BatchNormalization()

        self.mlp = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes)
        ])

    def call(self, inputs, training=False, mask=None):
        if len(inputs) == 4:
            x, edge_index, edge_weight, node_graph_index = inputs
        else:
            x, edge_index, _, node_graph_index = inputs
            edge_weight = None

        h1 = self.gin0([x, edge_index, edge_weight])
        h2 = self.bn1(h1)
        h2 = self.gin1([h2, edge_index, edge_weight])
        h3 = self.bn1(h2)
        h3 = self.gin2([h3, edge_index, edge_weight])
        h4 = self.bn2(h3)
        h4 = self.gin3([h4, edge_index, edge_weight])
        h5 = self.bn3(h4)
        h5 = self.gin4([h5, edge_index, edge_weight])
        h5 = self.bn3(h5)

        h1 = tfg.nn.sum_pool(h1, node_graph_index)
        h2 = tfg.nn.sum_pool(h2, node_graph_index)
        h3 = tfg.nn.sum_pool(h3, node_graph_index)
        h4 = tfg.nn.sum_pool(h4, node_graph_index)
        h5 = tfg.nn.sum_pool(h5, node_graph_index)

        h = tf.concat((h1, h2, h3, h4, h5), axis=-1)
        out = self.mlp(h, training=training)

        return out


model = GIN(32)
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


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)

best_test_acc = 0
for step in range(0, 1000):
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
