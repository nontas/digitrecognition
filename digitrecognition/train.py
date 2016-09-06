import os
import shutil
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.python.platform import tf_logging as logging

from menpo.image import Image
from menpo.image import Image, BooleanImage
from menpo.feature import gaussian_filter
from menpo.shape import bounding_box
from menpo.transform import ThinPlateSplines
from menpo.transform import Similarity, Rotation, Translation, Affine

from digitrecognition.base import src_dir_path
import digitrecognition.params

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE = 'test.tfrecords'

FLAGS = tf.app.flags.FLAGS


def baseline(images):
    net = slim.layers.conv2d(images, 32, 5, scope='conv1')
    net = slim.layers.max_pool2d(net, 2, scope='pool1')
    net = slim.layers.conv2d(net, 64, 5, scope='conv2')
    net = slim.layers.max_pool2d(net, 2, scope='pool2')
    net = slim.layers.flatten(net, scope='flatten3')
    net = slim.layers.fully_connected(net, 500, scope='fc4')
    net = slim.layers.fully_connected(net, 10, activation_fn=None,
                                      scope='fc5')
    return net


def ultimate(images):
    with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected], normalizer_fn=slim.batch_norm):
        net = slim.layers.conv2d(images, 64, 5, scope='conv1')
        net = slim.layers.max_pool2d(net, 2, scope='pool1')
        net = slim.layers.conv2d(net, 32, 5, scope='conv2')
        net = slim.layers.max_pool2d(net, 2, scope='pool2')
        net = slim.layers.flatten(net, scope='flatten3')
        net = slim.layers.dropout(net, scope='dropout4')
        net = slim.layers.fully_connected(net, 1024, scope='fc5')
    net = slim.layers.fully_connected(net, 10, activation_fn=None, scope='fc6')
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
    image = tf.reshape(image, [mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def skew(shape, theta, phi):
    rotate_ccw = Similarity.init_identity(shape.n_dims)
    h_matrix = np.ones((3, 3))
    h_matrix[0, 1] = np.tan(theta * np.pi / 180.)
    h_matrix[1, 0] = np.tan(phi * np.pi / 180.)
    h_matrix[:2, 2] = 0.
    h_matrix[2, :2] = 0.
    r = Affine(h_matrix)
    t = Translation(-shape.centre(), skip_checks=True)
    # Translate to origin, rotate counter-clockwise, then translate back
    rotate_ccw.compose_before_inplace(t)
    rotate_ccw.compose_before_inplace(r)
    rotate_ccw.compose_before_inplace(t.pseudoinverse())

    return rotate_ccw.apply(shape)


def preprocess(pixels, low=-30, high=30):
    # Create menpo image
    image = Image.init_from_channels_at_back(pixels)

    # Rotation
    theta = np.random.uniform(low=low, high=high)
    image = image.rotate_ccw_about_centre(theta, retain_shape=True)

    # Skew
    mask = BooleanImage(image.pixels[0])
    bbox = bounding_box(*mask.bounds_true())
    angles = np.random.uniform(low=low, high=high, size=2)
    new_bb = skew(bbox, angles[0], angles[1])
    pwa = ThinPlateSplines(new_bb, bbox)
    image = image.warp_to_shape(image.shape, pwa)
    
    return image.pixels_with_channels_at_back()[..., None].astype(np.float32)


def inputs(set_name, batch_size, num_epochs=None, one_hot_labels=False):
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

    set_names = {
        'train': TRAIN_FILE,
        'validation': VALIDATION_FILE,
        'test': TEST_FILE
    }

    if set_name not in set_names:
        raise RuntimeError('{} is not a valid set name'.format(set_name))
    
    filename = train_dir / set_names[set_name]

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [str(filename)], num_epochs=num_epochs, shuffle=False)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue)
        
        # Augment
        if set_name == 'train':
            shape = image.get_shape()
            image, = tf.py_func(preprocess, [image], [tf.float32])
            image.set_shape(shape)
        
        image = image / 255.0 - 0.5

        if one_hot_labels:
            label = tf.one_hot(label, mnist.NUM_CLASSES, dtype=tf.int32)

        num_threads = 4 if set_name == 'train' else 1

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.

        if set_name == 'train':
            images, sparse_labels = tf.train.shuffle_batch(
                    [image, label],
                    batch_size=batch_size,
                    num_threads=num_threads,
                    capacity=10000,
                    min_after_dequeue=batch_size*4
            )
        else:
            images, sparse_labels = tf.train.batch([image, label], batch_size=batch_size)

    return images, sparse_labels


def train(batch_size, num_batches, initial_learning_rate, decay_steps,
          decay_rate, optimization, momentum, log_dir, verbose=False):
    # Reset graph nodes
    tf.reset_default_graph()

    # Delete log
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    # Define network
    images, labels = inputs(set_name='train', batch_size=batch_size,
                            num_epochs=num_batches, one_hot_labels=True)

    with slim.arg_scope([slim.layers.dropout, slim.batch_norm], is_training=True):
#        with tf.device('/gpu:0'):
        predictions = ultimate(images)

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
    slim.learning.train(train_op, log_dir, save_summaries_secs=20, save_interval_secs=60, 
                        log_every_n_steps=100)


def main(argv):
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
    tf.app.run(main)
