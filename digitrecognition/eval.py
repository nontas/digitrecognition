import os
import shutil
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import tf_logging as logging

from digitrecognition.train import inputs
from digitrecognition.model import get_network

# This is necessary in order to get the parameters from the argparser
import digitrecognition.params
FLAGS = tf.app.flags.FLAGS


def evaluate(architecture, batch_size, num_samples, log_dir, checkpoint_dir,
             set_name, verbose=False):
    r"""
    Method that evaluates a network.

    Parameters
    ----------
    architecture : ``{'baseline', 'ultimate', 'ultimate_v2'}``
        The network architecture to use.
    batch_size : `int`
        The number of images per batch.
    num_samples : `int`
        The number of samples.
    log_dir : `str`
        The log directory (the log data are used by tensorboard).
    checkpoint_dir : `str`
        The directory that contains the saved checkpoints. This is normally the
        log dir of training.
    set_name : ``{'train', 'validation', 'test'}``
        The dataset on which the evaluation will be performed.
    verbose : `bool`, optional
        If `True`, then the log will be printed on the terminal.
    """
    # Delete log
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    # Define queue data provider functionality
    images, labels = inputs(set_name=set_name, batch_size=batch_size,
                            num_epochs=None)

    # Define network
    with slim.arg_scope([slim.layers.dropout, slim.batch_norm],
                        is_training=False):
        net_fun = get_network(architecture)
        predictions = net_fun(images)

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

    # Verbose
    if verbose:
        logging.set_verbosity(1)

    # Find optimal number of batches (evaluations)
    num_batches = math.ceil(num_samples / float(batch_size))

    # Run evaluation loop
    slim.evaluation.evaluation_loop(
        '', checkpoint_dir, log_dir, num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        summary_op=tf.merge_all_summaries(), eval_interval_secs=300)


def main(_):
    evaluate(FLAGS.architecture,
             FLAGS.batch_size,
             FLAGS.num_samples,
             FLAGS.log_eval_dir,
             FLAGS.log_train_dir,
             FLAGS.eval_set,
             FLAGS.verbose)


if __name__ == '__main__':
    tf.app.run()
