import os
import shutil

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import tf_logging as logging

from digitrecognition.data_provider import inputs
from digitrecognition.model import get_network

# This is necessary in order to get the parameters from the argparser
import digitrecognition.params
FLAGS = tf.app.flags.FLAGS


def train(architecture, batch_size, num_batches, initial_learning_rate,
          decay_steps, decay_rate, optimization, momentum, log_dir,
          verbose=False):
    r"""
    Method that trains a network.

    Parameters
    ----------
    architecture : ``{'baseline', 'ultimate', 'ultimate_v2'}``
        The network architecture to use.
    batch_size : `int`
        The number of images per batch.
    num_batches : `int` or ``None``, optional
        The number of batches.
    initial_learning_rate : `float`
        The initial value of the learning rate decay.
    decay_steps : `float`
        The step of the learning rate decay.
    decay_rate : `float`
        The rate of the learning rate decay.
    optimization : ``{'rms', 'adam'}``
        The optimization technique to use.
    momentum : `float`
        The momentum in case the RMS optimization is used.
    log_dir : `str`
        The log directory (the log data are used by tensorboard).
    verbose : `bool`, optional
        If `True`, then the log will be printed on the terminal.
    """
    # Reset graph nodes
    tf.reset_default_graph()

    # Delete log
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    # Define queue data provider functionality
    images, labels = inputs(set_name='train', batch_size=batch_size,
                            num_epochs=num_batches, one_hot_labels=True)

    # Define network
    with slim.arg_scope([slim.layers.dropout, slim.batch_norm],
                        is_training=True):
        # with tf.device('/gpu:0'):
        net_fun = get_network(architecture)
        predictions = net_fun(images)

    # Display images to tensorboard
    tf.image_summary('images', images, max_images=5)

    # Define loss function
    slim.losses.softmax_cross_entropy(predictions, labels)
    total_loss = slim.losses.get_total_loss()

    # Add loss graph to tensorboard
    tf.scalar_summary('loss', total_loss)

    # Create learning rate decay
    global_step = slim.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step=global_step, decay_steps=decay_steps,
        decay_rate=decay_rate)

    # Create optimizer
    if optimization == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9,
                                              momentum=momentum)
    elif optimization == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError('Optimization must be either rms or adam.')

    # Create training operation
    train_op = slim.learning.create_train_op(total_loss, optimizer,
                                             summarize_gradients=True)

    # Verbose
    if verbose:
        logging.set_verbosity(1)

    # Start training
    slim.learning.train(train_op, log_dir, save_summaries_secs=20,
                        save_interval_secs=60, log_every_n_steps=100)


def main(argv):
    train(FLAGS.architecture,
          FLAGS.batch_size,
          FLAGS.num_train_batches,
          FLAGS.initial_learning_rate,
          FLAGS.decay_steps,
          FLAGS.decay_rate,
          FLAGS.optimization,
          FLAGS.momentum,
          FLAGS.log_train_dir,
          FLAGS.verbose)


if __name__ == '__main__':
    tf.app.run(main)
