# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_geometric.utils import tf_utils
import tf_geometric as tfg
import tensorflow as tf

graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()

num_classes = graph.y.max() + 1
drop_rate = 0.6
checkpoint_dir = "./models"
checkpoint_prefix = os.path.join(checkpoint_dir, "gat")


# Multi-layer GAT Model
class GATModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gat0 = tfg.layers.GAT(64, activation=tf.nn.relu, num_heads=8, drop_rate=drop_rate, attention_units=8)
        self.gat1 = tfg.layers.GAT(num_classes, drop_rate=drop_rate, attention_units=1)
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs
        h = self.dropout(x, training=training)
        h = self.gat0([h, edge_index], training=training)
        h = self.dropout(h, training=training)
        h = self.gat1([h, edge_index], training=training)
        return h


# Model/Layer objects in TensorFlow may delay the creation of variables to their first call, when input shapes are available.
# Therefore, you must call the model at least once before writing checkpoints.
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
def evaluate():
    logits = forward(graph)
    masked_logits = tf.gather(logits, test_index)
    masked_labels = tf.gather(graph.y, test_index)
    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    corrects = tf.equal(y_pred, masked_labels)
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)


@tf_utils.function
def train_step():
    with tf.GradientTape() as tape:
        logits = forward(graph, training=True)
        loss = compute_loss(logits, train_index, tape.watched_variables())

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    return loss


# tf.tain.Checkpoint can save and restore trackable objects.
# You can pass trackable objects as keywords arguments as follows:
# tf.train.Checkpoint(key1=value1, key2=value2, ...)
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

for step in range(1, 401):

    loss = train_step()

    if step % 20 == 0:
        accuracy = evaluate()
        print("step = {}\tloss = {}\taccuracy = {}".format(step, loss, accuracy))

        # write checkpoints
        checkpoint.save(file_prefix=checkpoint_prefix)
        print("write checkpoint at step {}".format(step))

# create new model and restore it from the checkpoint
restored_model = GATModel()
# if you want to restore the optimizer, just add it as a keyword argument as follows:
# checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint = tf.train.Checkpoint(model=restored_model)

# https://www.tensorflow.org/guide/checkpoint#delayed_restorations
# Layer/Model objects in TensorFlow may delay the creation of variables to their first call, when input shapes are available.
# For example the shape of a Dense layer's kernel depends on both the layer's input and output shapes,
# and so the output shape required as a constructor argument is not enough information to create the variable on its own.
# Since calling a Layer/Model also reads the variable's value, a restore must happen between the variable's creation and its first use.
# To support this idiom, tf.train.Checkpoint queues restores which don't yet have a matching variable.
# In this case, some variables, such as model.gat0.kernel and model.gat0.bias will not be immediately restored after calling checkpoint.restore.
# The will be automatically restored during the first call of restored_model.
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# @tf_utils.function can speed up functions for TensorFlow 2.x
@tf_utils.function
def forward_by_restored_model(graph, training=False):
    return restored_model([graph.x, graph.edge_index], training=training)


print("\ninfer with model:")
print(forward(graph))

print("\ninfer with restored_model:")
print(forward_by_restored_model(graph))
