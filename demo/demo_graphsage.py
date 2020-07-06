# coding=utf-8
import os
import tensorflow as tf
from tensorflow import keras
from tf_geometric.layers.conv.graphSAGE import GraphSAGE
from tf_geometric.datasets.cora import CoraDataset

from tf_geometric.utils.neighbor_sample import get_neighbors, sample_neighbors

# tf.random.set_seed(-1)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

graph, (train_index, valid_index, test_index) = CoraDataset().load_data()

num_classes = graph.y.max() + 1

model = GraphSAGE(128, num_classes, aggregate_type='max_pooling', drop_rate=0.5)

def forward(graph, num_layers, sizes=None):

    ##sample neighbors
    # sizes = [10,5,5]
    edge_index_list = []
    edge_weight_list = []
    try:
        if not sizes is None and num_layers == len(sizes):
            to_neighbors = get_neighbors(graph.edge_index)

            for size in sizes:
                sampled_edge_index, sampled_edge_weight = sample_neighbors(to_neighbors, graph.edge_index,
                                                                           graph.edge_weight, num_sample=size)
                edge_index_list.append(sampled_edge_index)
                edge_weight_list.append(sampled_edge_weight)
        elif sizes is None:
            to_neighbors = get_neighbors(graph.edge_index)

            num_sample = 0
            for neighs in to_neighbors:
                num_sample = max(num_sample, len(neighs))

            for _ in range(num_layers):
                sampled_edge_index, sampled_edge_weight = sample_neighbors(to_neighbors, graph.edge_index,
                                                                           graph.edge_weight, num_sample=num_sample)
                edge_index_list.append(sampled_edge_index)
                edge_weight_list.append(sampled_edge_weight)


        h = model([graph.x, edge_index_list, edge_weight_list])

        return h

    except:
        print("the size of sampled neighbor nodes does not match the number of layers in graphSAGE.")



def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vals = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4


def evaluate():
    logits = forward(graph, num_layers=3)
    masked_logits = tf.gather(logits, test_index)
    masked_labels = tf.gather(graph.y, test_index)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    accuracy_m = keras.metrics.Accuracy()
    accuracy_m.update_state(masked_labels, y_pred)
    return accuracy_m.result().numpy()


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

for step in range(600):
    with tf.GradientTape() as tape:
        logits = forward(graph, num_layers=3, sizes=[10,5,3])
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))