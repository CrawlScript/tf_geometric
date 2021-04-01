# coding=utf-8
import os

# multi-gpu ids
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import tf_geometric as tfg
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

batch_size = 1024
drop_rate = 0.4

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
            yield (batch_graph.x, batch_graph.edge_index, batch_graph.node_graph_index), batch_graph.y

        if not infinite:
            break


class MeanPoolNetwork(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gcn0 = tfg.layers.GCN(256, activation=tf.nn.relu)
        self.gcn1 = tfg.layers.GCN(256, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        x, edge_index, node_graph_index = inputs

        # bug fix for distributed training
        node_graph_index = tf.reshape(node_graph_index, [-1])

        # GCN Encoder
        h = self.gcn0([x, edge_index], training=training)
        h = self.dropout(h, training=training)
        h = self.gcn1([h, edge_index], training=training)

        # Mean Pooling
        h = tfg.nn.mean_pool(h, node_graph_index)
        h = self.dropout(h, training=training)

        # Predict Graph Labels
        h = self.dense(h)
        return h


strategy = tf.distribute.MirroredStrategy()


def train_dataset_fn(ctx):
    def create_replica_train_generator():
        return create_graph_generator(train_graphs, batch_size // strategy.num_replicas_in_sync, infinite=True,
                                      shuffle=True)

    return tf.data.Dataset.from_generator(
        create_replica_train_generator,
        output_types=((tf.float32, tf.int32, tf.int32), tf.int32),
        output_shapes=(
        (tf.TensorShape([None, graphs[0].x.shape[1]]), tf.TensorShape([2, None]), tf.TensorShape([None])),
        tf.TensorShape([None])
        )
    )


distributed_train_dataset = strategy.experimental_distribute_datasets_from_function(train_dataset_fn)

# The model will automatically use all seen GPUs defined by "CUDA_VISIBLE_DEVICES" for distributed training
with strategy.scope():
    model = MeanPoolNetwork()


def cross_entropy(y_true, logits):
    y_true = tf.cast(y_true, tf.int32)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=tf.one_hot(y_true, depth=num_classes)
    )
    return tf.reduce_mean(losses)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3),
    loss=cross_entropy
)


def forward(batch_graph, training=False):
    return model([batch_graph.x, batch_graph.edge_index, batch_graph.node_graph_index], training=training)


def evaluate():
    corrects = []

    for (x, edge_index, node_graph_index), y in create_graph_generator(test_graphs, batch_size, shuffle=False,
                                                                       infinite=False):
        logits = model([x, edge_index, node_graph_index])
        preds = tf.argmax(logits, axis=-1)
        corrects.append(tf.equal(preds, y))

    corrects = tf.concat(corrects, axis=0)
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

    return accuracy


class EvaluationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            test_accuracy = evaluate()
            print("\nepoch = {}\ttest_accuracy = {}".format(epoch, test_accuracy))


# The model will automatically use all seen GPUs defined by "CUDA_VISIBLE_DEVICES" for distributed training
model.fit(distributed_train_dataset, steps_per_epoch=len(graphs) // batch_size, epochs=201,
          callbacks=[EvaluationCallback()], verbose=1)

accuracy = evaluate()
print("final test_accuracy = {}".format(accuracy))
