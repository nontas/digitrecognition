import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist

from menpo.image import Image, BooleanImage
from menpo.transform import Similarity, Translation, Affine, ThinPlateSplines
from menpo.shape import bounding_box

from digitrecognition.base import src_dir_path


TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE = 'test.tfrecords'


def skew_shape(pointcloud, theta, phi):
    r"""
    Method that skews the provided pointcloud.

    Parameters
    ----------
    pointcloud : `menpo.shape.PointCloud`
        The shape to distort.
    theta : `float`
        The skew angle over x axis (tan(theta)).
    phi : `float`
        The skew angle over y axis (tan(phi)).

    Returns
    -------
    skewed_shape : `menpo.shape.PointCloud`
        The skewed (distorted) pointcloud.
    """
    rotate_ccw = Similarity.init_identity(pointcloud.n_dims)
    # Create skew matrix
    h_matrix = np.ones((3, 3))
    h_matrix[0, 1] = np.tan(theta * np.pi / 180.)
    h_matrix[1, 0] = np.tan(phi * np.pi / 180.)
    h_matrix[:2, 2] = 0.
    h_matrix[2, :2] = 0.
    r = Affine(h_matrix)
    t = Translation(-pointcloud.centre(), skip_checks=True)
    # Translate to origin, rotate counter-clockwise, then translate back
    rotate_ccw.compose_before_inplace(t)
    rotate_ccw.compose_before_inplace(r)
    rotate_ccw.compose_before_inplace(t.pseudoinverse())

    return rotate_ccw.apply(pointcloud)


def skew_image(image, theta, phi):
    r"""
    Method that skews the provided image. Note that the output image has the
    same size (shape) as the input.

    Parameters
    ----------
    image : `menpo.image.Image`
        The image to distort.
    theta : `float`
        The skew angle over x axis (tan(theta)).
    phi : `float`
        The skew angle over y axis (tan(phi)).

    Returns
    -------
    skewed_image : `menpo.image.Image`
        The skewed (distorted) image.
    """
    # Get mask of pixels
    mask = BooleanImage(image.pixels[0])
    # Create the bounding box (pointcloud) of the mask
    bbox = bounding_box(*mask.bounds_true())
    # Skew the bounding box
    new_bbox = skew_shape(bbox, theta, phi)
    # Warp the image using TPS
    pwa = ThinPlateSplines(new_bbox, bbox)

    return image.warp_to_shape(image.shape, pwa)


def preprocess(pixels, min_angle=-30, max_angle=30):
    r"""
    Method that applies some pre-processing on the provided image. This involves:

        1. Rotating the image about its centre with random angle.
        2. Skewing the image with random variables.

    The random variables of the two transforms are generated from a uniform
    distribution.

    Parameters
    ----------
    pixels : `array`
        The input image of shape `(height, width, n_channels)`.
    min_angle : `int`, optional
        The minimum angle value of the uniform distribution.
    max_angle : `int`
        The maximum angle value of the uniform distribution.

    Returns
    -------
    transformed_image : `array`
        The transformed image of shape `(height, width, n_channels)`.
    """
    # Create menpo image
    image = Image.init_from_channels_at_back(pixels)

    # Rotation
    theta = np.random.uniform(low=min_angle, high=max_angle)
    image = image.rotate_ccw_about_centre(theta, retain_shape=True)

    # Skew
    angles = np.random.uniform(low=min_angle, high=max_angle, size=2)
    image = skew_image(image, angles[0], angles[1])

    return image.pixels_with_channels_at_back()[..., None].astype(np.float32)


def read_and_decode(filename_queue):
    r"""
    Method that reads and decodes an image and its label from the data queue.

    Parameters
    ----------
    filename_queue : `queue`
        The files queue.

    Returns
    -------
    image : `tf.Tensor`
        The loaded image.
    label : `tf.int32`
        The corresponding label.
    """
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

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def inputs(set_name, batch_size, num_epochs=None, one_hot_labels=False):
    r"""
    Method that reads the input data num_epochs times. Note that if the data are
    loaded from the training set, then a pre-processing step is applied which
    first rotates and then skews the image. Also, note that an
    `tf.train.QueueRunner` is added to the graph, which must be run using
    e.g. `tf.train.start_queue_runners()`.

    Parameters
    ----------
    set_name : ``{'train', 'validation', 'test'}``
        The dataset from which to load the data.
    batch_size : `int`
        The number of images per returned batch.
    num_epochs : `int` or ``None``, optional
        The number of times to read the input data. If None, the data will be
        read forever.
    one_hot_labels : `bool`, optional
        Whether to return the label in the one-hot vector format.

    Returns
    -------
    images : `tf.Tensor`
        A float tensor with shape `(n_images, height, width, n_channels)`.
    sparse_labels : `tf.Tensor`
        An int32 tensor with shape `(n_images, )` with the label per image.
    """
    # Check if dataset exists.
    set_names = {
        'train': TRAIN_FILE,
        'validation': VALIDATION_FILE,
        'test': TEST_FILE
    }
    if set_name not in set_names:
        raise RuntimeError('{} is not a valid set name'.format(set_name))

    # Get filename of dataset
    filename = src_dir_path() / 'data' / set_names[set_name]

    # Create filename queue and read
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [str(filename)], num_epochs=num_epochs, shuffle=False)

        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue)

        # Pre-process the data (rotation, skew), if we are loading from the
        # trainig set.
        if set_name == 'train':
            shape = image.get_shape()
            image, = tf.py_func(preprocess, [image], [tf.float32])
            image.set_shape(shape)

        # Normalize pixel values to be in range [-0.5, 0.5].
        image = tf.to_float(image) / 255.0 - 0.5

        # Convert label to one-hot vector, if required.
        if one_hot_labels:
            label = tf.one_hot(label, mnist.NUM_CLASSES, dtype=tf.int32)

        # If loading the training set, then use 4 threads.
        num_threads = 2 if set_name == 'train' else 1

        # If training set, then shuffle the images and collect them into
        # batch_size batches.
        if set_name == 'train':
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=num_threads,
                capacity=10000, min_after_dequeue=batch_size*4)
        else:
            images, sparse_labels = tf.train.batch([image, label],
                                                   batch_size=batch_size)

    return images, sparse_labels
