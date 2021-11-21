# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_geometric.utils import tf_utils
import tensorflow as tf
import tf_geometric as tfg
from tqdm import tqdm
import time

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()

num_classes = graph.y.max() + 1
drop_rate = 0.5
learning_rate = 5e-3
l2_coef = 1e-3


# SSGC Model
class SSGCModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssgc = tfg.layers.SSGC(k=10, alpha=0.1, units_list=[64, num_classes],
                                    edge_drop_rate=0.0, prop_feature_drop_rate=drop_rate, dense_drop_rate=drop_rate)
        self.dropout = tf.keras.layers.Dropout(0.0)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index, edge_weight = inputs
        h = self.dropout(x, training=training)
        h = self.ssgc([h, edge_index, edge_weight], training=training, cache=cache)
        return h


model = SSGCModel()


# @tf_utils.function can speed up functions for TensorFlow 2.x.
# @tf_utils.function is not compatible with TensorFlow 1.x and dynamic graph.cache.
@tf_utils.function
def forward(graph, training=False):
    return model([graph.x, graph.edge_index, graph.edge_weight], training=training, cache=graph.cache)


# The following line is only necessary for using SSGC with @tf_utils.function
# For usage without @tf_utils.function, you can commont the following line and SSGC layers can automatically manager the cache
model.ssgc.build_cache_for_graph(graph)


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

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * l2_coef


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


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
