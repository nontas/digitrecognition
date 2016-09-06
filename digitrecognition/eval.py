import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist

from digitrecognition.train import inputs, ultimate
import digitrecognition.params
import math

FLAGS = tf.app.flags.FLAGS


def evaluate(batch_size, num_samples, log_dir, checkpoint_dir, set_name):
    # Delete log
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    # Define network
    images, labels = inputs(set_name=set_name, batch_size=batch_size, num_epochs=None)
    with slim.arg_scope([slim.layers.dropout, slim.batch_norm], is_training=False):
        predictions = ultimate(images)

    # Convert predictions to numbers
    predictions = tf.to_int32(tf.argmax(predictions, 1))

    # Report the streaming accuracy (running accuracy)
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map(
     {
        "accuracy": slim.metrics.streaming_accuracy(predictions, labels)
     })

    # Define the streaming summaries for tensorboard
    for metric_name, metric_value in metrics_to_values.items():
        tf.scalar_summary(metric_name, metric_value)

    num_batches = math.ceil(num_samples / float(batch_size))
    # Evaluate every 30 seconds
    slim.evaluation.evaluation_loop(
        '', checkpoint_dir, log_dir, num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        summary_op=tf.merge_all_summaries(),
        eval_interval_secs=60)


def main(_):
    evaluate(FLAGS.batch_size,
             FLAGS.num_samples,
             FLAGS.log_eval_dir,
             FLAGS.log_train_dir,
             FLAGS.eval_set)


if __name__ == '__main__':
    tf.app.run()
