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
num_features = train_graphs[0].x.shape[-1]

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


def sample_edge_index_list(graph):
    sampled_edge_index_list = []
    neighbor_sampler = graph.cache["sampler"]
    for num_sampled_neighbors in num_sampled_neighbors_list:
        sampled_edge_index, _ = neighbor_sampler.sample(k=num_sampled_neighbors)
        sampled_edge_index_list.append(sampled_edge_index)
    return sampled_edge_index_list


@tf.function(
    input_signature=(
        train_graphs[0].tensor_spec_x,
        tuple([tfg.Graph.tensor_spec_edge_index for _ in graph_sages]),
        tf.TensorSpec(shape=[], dtype=tf.bool)
    )
)
def forward(x, sampled_edge_index_list, training=False):
    h = x
    for i, (graph_sage, sampled_edge_index) in enumerate(zip(graph_sages, sampled_edge_index_list)):
        h = graph_sage([h, sampled_edge_index], training=training)
    h = fc(h, training=training)
    return h


def compute_loss(logits, y, vars):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=tf.cast(y, tf.float32)
    )

    kernel_vars = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vars]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 1e-5


def calc_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0

    return f1_score(y_true, y_pred, average="micro")


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)



@tf.function(
    input_signature=(
        train_graphs[0].tensor_spec_x,
        tuple([tfg.Graph.tensor_spec_edge_index for _ in graph_sages]),
        train_graphs[0].tensor_spec_y
    )
)
def train_step(x, sampled_edge_index_list, y):
    with tf.GradientTape() as tape:
        logits = forward(x, sampled_edge_index_list, training=True)
        loss = compute_loss(logits, y, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    return loss


def evaluate(graphs):
    y_preds = []
    y_true = []

    for graph in graphs:
        y_true.append(graph.y)
        sampled_edge_index_list = sample_edge_index_list(graph)
        logits = forward(graph.x, sampled_edge_index_list)
        y_preds.append(logits.numpy())

    y_pred = np.concatenate(y_preds, axis=0)
    y = np.concatenate(y_true, axis=0)

    mic = calc_f1(y, y_pred)

    return mic


def create_generator():
    while True:
        for graph in train_graphs:

            sampled_edge_index_list = sample_edge_index_list(graph)

            x = tf.convert_to_tensor(graph.x)
            sampled_edge_index_list = tuple(
                [tf.convert_to_tensor(edge_index) for edge_index in sampled_edge_index_list])
            y = tf.convert_to_tensor(graph.y)

            yield x, sampled_edge_index_list, y


dataset = tf.data.Dataset.from_generator(
    create_generator,
    output_signature=(
        train_graphs[0].tensor_spec_x,
        tuple([tfg.Graph.tensor_spec_edge_index for _ in graph_sages]),
        train_graphs[0].tensor_spec_y
    )
).prefetch(20)


for step, (x, sampled_edge_index_list, y) in tqdm(enumerate(dataset)):
    loss = train_step(x, sampled_edge_index_list, y)

    if step % 100 == 0:
        valid_f1_mic = evaluate(valid_graphs)
        test_f1_mic = evaluate(test_graphs)
        print("step = {}\tloss = {}\tvalid_f1_micro = {}".format(step, loss, valid_f1_mic))
        print("step = {}\ttest_f1_micro = {}".format(step, test_f1_mic))

    if step == 1000:
        break
