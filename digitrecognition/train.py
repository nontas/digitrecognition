import os
import shutil

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.python.platform import tf_logging as logging

from digitrecognition.base import src_dir_path
import digitrecognition.params

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

FLAGS = tf.app.flags.FLAGS


def lenet(images):
    net = slim.layers.conv2d(images, 32, 5, scope='conv1')
    net = slim.layers.max_pool2d(net, 2, scope='pool1')
    net = slim.layers.conv2d(net, 64, 5, scope='conv2')
    net = slim.layers.max_pool2d(net, 2, scope='pool2')
    net = slim.layers.flatten(net, scope='flatten3')
    net = slim.layers.fully_connected(net, 500, scope='fully_connected4')
    net = slim.layers.fully_connected(net, 10, activation_fn=None,
                                      scope='fully_connected5')
    return net


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([mnist.IMAGE_PIXELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.reshape(image, [mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def inputs(train, batch_size, num_epochs=None, one_hot_labels=False):
    r"""
    Reads the input data num_epochs times.

    Parameters
    ----------
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or None to
        train forever.
    Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().
    """
    train_dir = src_dir_path() / 'data'
    if train:
        filename = train_dir / TRAIN_FILE
    else:
        filename = train_dir / VALIDATION_FILE

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [str(filename)], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue)

        if one_hot_labels:
            label = tf.one_hot(label, mnist.NUM_CLASSES, dtype=tf.int32)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

    return images, sparse_labels


def train(batch_size, num_batches, initial_learning_rate, decay_steps,
          decay_rate, optimization, momentum, log_dir, verbose=False):
    # Reset graph nodes
    tf.reset_default_graph()

    # Delete log
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    # Define network
    images, labels = inputs(train=True, batch_size=batch_size,
                            num_epochs=num_batches, one_hot_labels=True)
    predictions = lenet(images)

    # Display images to tensorboard
    tf.image_summary('images', images, max_images=5)

    # Define loss function
    slim.losses.softmax_cross_entropy(predictions, labels)
    total_loss = slim.losses.get_total_loss()

    # Add loss summary to tensorboard
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
    slim.learning.train(train_op, log_dir, save_summaries_secs=20)


def main(_):
    train(FLAGS.batch_size,
          FLAGS.num_train_batches,
          FLAGS.initial_learning_rate,
          FLAGS.decay_steps,
          FLAGS.decay_rate,
          FLAGS.optimization,
          FLAGS.momentum,
          FLAGS.log_train_dir,
          FLAGS.verbose)


if __name__ == '__main__':
    tf.app.run()
