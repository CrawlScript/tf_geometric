# coding=utf-8
import os
# multi-gpu ids
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4"
import tf_geometric as tfg
from tf_geometric.layers import GCN
from tensorflow.keras.regularizers import L1L2
import tensorflow as tf


graph, (train_index, valid_index, test_index) = tfg.datasets.CoraDataset().load_data()
num_classes = graph.y.max() + 1

drop_rate = 0.5
learning_rate = 1e-2
l2_coef = 5e-4


# custom network
class GCNNetwork(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcn0 = GCN(16, activation=tf.nn.relu, kernel_regularizer=L1L2(l2=l2_coef))
        self.gcn1 = GCN(num_classes, kernel_regularizer=L1L2(l2=l2_coef))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None):
        x, edge_index = inputs
        h = self.dropout(x, training=training)
        h = self.gcn0([h, edge_index], training=training)
        h = self.dropout(h, training=training)
        h = self.gcn1([h, edge_index], training=training)
        return h


# prepare a generator and a dataset for distributed training
def create_batch_generator():
    while True:
        yield (graph.x, graph.edge_index), graph.y


def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_generator(
        create_batch_generator,
        output_types=((tf.float32, tf.int32), tf.int32),
        output_shapes=((tf.TensorShape([None, graph.x.shape[1]]), tf.TensorShape([2, None])), tf.TensorShape([None]))
    )
    return dataset


strategy = tf.distribute.MirroredStrategy()
distributed_dataset = strategy.experimental_distribute_datasets_from_function(dataset_fn)

# The model will automatically use all seen GPUs defined by "CUDA_VISIBLE_DEVICES" for distributed training
with strategy.scope():
    model = GCNNetwork()


# custom loss function
def masked_cross_entropy(y_true, logits):
    y_true = tf.cast(y_true, tf.int32)
    masked_logits = tf.gather(logits, train_index)
    masked_labels = tf.gather(y_true, train_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    return tf.reduce_mean(losses)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=masked_cross_entropy,
    # run_eagerly=True
)


def evaluate():
    logits = model([graph.x, graph.edge_index])
    masked_logits = tf.gather(logits, test_index)
    masked_labels = tf.gather(graph.y, test_index)

    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)
    corrects = tf.cast(tf.equal(masked_labels, y_pred), tf.float32)
    accuracy = tf.reduce_mean(corrects)
    return accuracy.numpy()


class EvaluationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 20 == 0:
            test_accuracy = evaluate()
            print("epoch = {}\ttest_accuracy = {}".format(epoch, test_accuracy))


# The model will automatically use all seen GPUs defined by "CUDA_VISIBLE_DEVICES" for distributed training
model.fit(distributed_dataset, steps_per_epoch=1, epochs=201, callbacks=[EvaluationCallback()], verbose=2)


test_accuracy = evaluate()
print("final test_accuracy = {}".format(test_accuracy))