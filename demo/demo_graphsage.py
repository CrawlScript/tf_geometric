# coding=utf-8
import os
import tensorflow as tf
import tf_geometric as tfg
from tensorflow import keras
import numpy as np
from tf_geometric.layers.conv.graphsage import GraphSAGE
from tf_geometric.datasets.ppi import PPIDataset

from tf_geometric.utils.neighbor_sample import get_neighbors, sample_neighbors, sorted_edge_index
from sklearn.metrics import f1_score
from tqdm import tqdm
# tf.random.set_seed(-1)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_graphs, valid_graphs, test_graphs = PPIDataset().load_data()

for graphs in [train_graphs, valid_graphs, test_graphs]:
    for graph in graphs:
        to_neighbors = get_neighbors(graph.edge_index)
        graph.edge_index, graph.edge_weight = sorted_edge_index(graph, to_neighbors)

num_classes = train_graphs[0].y.shape[1]


aggregate_type = 'mean_pooling'
print('aggregate_type:',aggregate_type)

model = GraphSAGE(128, num_classes, aggregate_type=aggregate_type, drop_rate=0.5)

def forward(graph, num_layers, sizes=None, training=False):

    edge_index_list = []
    edge_weight_list = []

    if not sizes is None and num_layers == len(sizes):
        to_neighbors = get_neighbors(graph.edge_index)

        for size in sizes:
            sampled_edge_index, sampled_edge_weight = sample_neighbors(to_neighbors, graph.edge_index,
                                                                       graph.edge_weight, num_sample=size)
            edge_index_list.append(sampled_edge_index)
            edge_weight_list.append(sampled_edge_weight)
    elif sizes is None:

        for _ in range(num_layers):
            edge_index_list.append(graph.edge_index)
            edge_weight_list.append(graph.edge_weight)

    h = model([graph.x, edge_index_list, edge_weight_list], num_layers=num_layers, training=training)

    return h



def compute_loss(logits, graph, vars):
    losses = tf.compat.v1.losses.sigmoid_cross_entropy(
            multi_class_labels=graph.y,
            logits=logits
    )
    # losses = tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=tf.convert_to_tensor(graph.y, dtype=tf.float32),
    #     logits=logits
    #
    # )

    kernel_vals = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-5
def calc_f1(y_true, y_pred):

    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0

    return f1_score(y_true, y_pred, average="micro")

def evaluate(graphs, num_layers):
    y_preds = []
    y_true = []

    for batch_graphs in graphs:
        y_true.append(batch_graphs.y)
        logits = forward(batch_graphs, num_layers=num_layers)
        y_preds.append(logits.numpy())

    y_pred = np.concatenate(y_preds, axis=0)
    y = np.concatenate(y_true, axis=0)

    mic = calc_f1(y, y_pred)

    return mic


optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

np.random.shuffle(train_graphs)
num_layers = 2
for step in tqdm(range(10)):
    loss = 0
    for batch_graphs in train_graphs:
        with tf.GradientTape() as tape:
            logits = forward(batch_graphs, num_layers=num_layers, sizes=[25,10])
            loss = compute_loss(logits, batch_graphs, tape.watched_variables())

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

    valid_f1_mic = evaluate(valid_graphs, num_layers=num_layers)
    print("step = {}\tloss = {}\tvalid_f1_micro = {}".format(step, loss, valid_f1_mic))

test_f1_mic = evaluate(test_graphs, num_layers=num_layers)
print("\ttest_f1_micro = {}".format( test_f1_mic))