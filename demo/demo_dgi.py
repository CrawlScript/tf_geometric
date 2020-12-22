# coding=utf-8
import os

from tf_geometric.utils import tf_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()
num_classes = graph.y.max() + 1

embedding_size = 512
drop_rate = 0.0
dropout = keras.layers.Dropout(drop_rate)


class GCNNetwork(keras.Model):
    """
    1-layer GCN model
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn = tfg.layers.GCN(
            embedding_size,
            activation=tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(value=0.25))
        )

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs
        h = self.gcn([x, edge_index], cache=cache)
        return h


class Bilinear(keras.Model):
    """
    Bilinear Model for DGL loss
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense = None
        self.bias = tf.Variable(0.0)

    def build(self, input_shapes):
        self.dense = tf.keras.layers.Dense(input_shapes[1][-1], use_bias=False)

    def call(self, inputs, training=None, mask=None, cache=None):
        a, b = inputs
        h = tf.reduce_sum(self.dense(a) * b, axis=-1) + self.bias
        return h


# GCN-based Encoder
model = GCNNetwork()

# Bilinear Model for DGL loss
bilinear_model = Bilinear()


def encode(graph, permutation=False, training=False):
    # permute nodes to create a noisy graph for negative sampling
    if permutation:
        perm_index = np.random.permutation(graph.x.shape[0])
        x = tf.gather(graph.x, perm_index)
    else:
        x = graph.x
    h = model([x, graph.edge_index], cache=graph.cache, training=training)
    return h


# Fast evaluation with sklearn
def evaluate_with_sklearn():
    embedded = encode(graph)
    embeddings = embedded.numpy()
    train_X, train_Y, test_X, test_Y = embeddings[train_index], graph.y[train_index], embeddings[test_index], graph.y[test_index]
    cls = LogisticRegression(C=10000, max_iter=500)
    cls.fit(train_X, train_Y)
    pred_Y = cls.predict(test_X)
    micro_f1 = f1_score(test_Y, pred_Y, average="micro")
    return micro_f1


# Following https://github.com/PetarV-/DGI, we train a full-connected layer for evaluation, which is slow
def evaluate_with_tf_keras():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2))
    embedded = encode(graph)
    embeddings = embedded.numpy()
    train_X, train_Y, test_X, test_Y = embeddings[train_index], graph.y[train_index], embeddings[test_index], graph.y[test_index]
    model.fit(train_X, tf.one_hot(train_Y, depth=num_classes).numpy(), epochs=100, verbose=False)
    pred_Y = tf.argmax(model(test_X), axis=-1)
    micro_f1 = f1_score(test_Y, pred_Y, average="micro")
    return micro_f1


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for step in range(301):
    with tf.GradientTape() as tape:
        # positive node representations
        pos_h = encode(graph, permutation=False, training=True)
        # negative node representations
        neg_h = encode(graph, permutation=True, training=True)
        # positive graph representations
        pos_graph_h = tf.nn.sigmoid(tf.reduce_mean(pos_h, axis=0, keepdims=True))

        pos_logits = bilinear_model([pos_h, pos_graph_h], training=True)
        neg_logits = bilinear_model([neg_h, pos_graph_h], training=True)

        pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pos_logits,
            labels=tf.ones_like(pos_logits)
        )

        neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=neg_logits,
            labels=tf.zeros_like(neg_logits)
        )

        # DGI loss
        loss = tf.reduce_mean(pos_losses + neg_losses)

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 50 == 0:
        micro_f1 = evaluate_with_sklearn()
        print("step = {}\tloss = {}\tmicro_f1 = {}".format(step, loss, micro_f1))

print("start final evaluation with tf.keras ......")
micro_f1 = evaluate_with_tf_keras()
print("final micro_f1 = {}".format(micro_f1))
