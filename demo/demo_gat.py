# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_geometric.utils import tf_utils
import tf_geometric as tfg
import tensorflow as tf
import time
from tqdm import tqdm

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()

num_classes = graph.y.max() + 1
drop_rate = 0.6


# Multi-layer GAT Model
class GATModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gat0 = tfg.layers.GAT(64, activation=tf.nn.relu, num_heads=8, attention_units=8, edge_drop_rate=drop_rate)
        self.gat1 = tfg.layers.GAT(num_classes, num_heads=1, attention_units=1, edge_drop_rate=drop_rate)

        # The GAT paper mentioned that: "Specially, if we perform multi-head attention on the final (prediction) layer of
        # the network, concatenation is no longer sensible - instead, we employ averaging".
        # In tf_geometric, if you want to set num_heads > 1 for the last output GAT layer, you can set split_value_heads=False
        # as follows to employ averaging instead of concatenation.
        # self.gat1 = tfg.layers.GAT(num_classes, num_heads=8, attention_units=8, split_value_heads=False, edge_drop_rate=drop_rate)

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs
        h = self.dropout(x, training=training)
        h = self.gat0([h, edge_index], training=training)
        h = self.dropout(h, training=training)
        h = self.gat1([h, edge_index], training=training)
        return h


model = GATModel()


# @tf_utils.function can speed up functions for TensorFlow 2.x
@tf_utils.function
def forward(graph, training=False):
    return model([graph.x, graph.edge_index], training=training)


@tf_utils.function
def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    kernel_vars = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vars]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4


@tf_utils.function
def train_step():
    with tf.GradientTape() as tape:
        logits = forward(graph, training=True)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return loss


@tf_utils.function
def evaluate():
    logits = forward(graph)
    masked_logits = tf.gather(logits, test_index)
    masked_labels = tf.gather(graph.y, test_index)
    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    corrects = tf.equal(y_pred, masked_labels)
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

for step in range(1, 401):
    loss = train_step()

    if step % 20 == 0:
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

print("\nstart speed test...")
num_test_iterations = 1000
start_time = time.time()
for _ in tqdm(range(num_test_iterations)):
    logits = forward(graph)
end_time = time.time()
print("mean forward time: {} seconds".format((end_time - start_time) / num_test_iterations))

if tf.__version__[0] == "1":
    print("** @tf_utils.function is disabled in TensorFlow 1.x. "
          "Upgrade to TensorFlow 2.x for 10X faster speed. **")
