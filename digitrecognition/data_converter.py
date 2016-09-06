import os
import urllib
import gzip
import numpy as np
import tensorflow as tf

from menpo.image import Image
from menpo.visualize import print_dynamic

from digitrecognition.base import src_dir_path


# MNIST url
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

# MNIST filenames
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


def download(filename, verbose=False):
    r"""
    Method that downloads the provided filename from SOURCE_URL and
    stores it in the data path, if it doesn't already exist.

    Parameters
    ----------
    filename : `str`
        The filename to download.
    verbose : `bool`, optional
        If `True`, then the progress will be printed.

    Returns
    -------
    file_path : `pathlib.PosixPath`
        The path where the file was stored.
    """
    if verbose:
        print_dynamic('Downloading {}'.format(filename))
    # Path to store data
    data_path = src_dir_path() / 'data'
    # Check if data path exists, otherwise create it
    if not os.path.isdir(str(data_path)):
        os.makedirs(str(data_path))
    # Check if file exists
    file_path = data_path / filename
    if not os.path.isfile(str(file_path)):
        # It doesn't exist, so download it
        urllib.request.urlretrieve(SOURCE_URL + filename,
                                   filename=str(file_path))
    # Return the path where the file is stored
    return file_path


def _read32(bytestream):
    r"""
    Read bytes as 32-bit integers.

    Parameters
    ----------
    bytestream : `bytes`
        The bytes to read.

    Returns
    -------
    array : `array`
        The 32-bit int data.
    """
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename, as_images=False, verbose=False):
    r"""
    Extract images from gzip file.

    Parameters
    ----------
    filename : `pathlib.PosixPath`
        The gzip file path.
    as_images : `bool`, optional
        If `True`, then the method returns a list containing a
        `menpo.image.Image` object per image. If `False`, then it
        returns a numpy array of shape `(n_images, height, width, n_channels)`.
    verbose : `bool`, optional
        If `True`, then the progress will be printed.

    Returns
    -------
    images : `list` or `array`
        The image data.
    """
    if verbose:
        print_dynamic('Extracting {}'.format(filename))
    with open(str(filename), 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        # Convert data array to list of menpo.image.Image, if required
        if as_images:
            return [Image(data[i, :, :, 0]) for i in range(data.shape[0])]
        return data


def _convert_dense_to_one_hot(labels_dense):
    r"""
    Method that converts an array of labels to one-hot labels.

    Parameters
    ----------
    labels_dense : `array`
        An `(n_images,)` array with an integer label per image.

    Returns
    -------
    labels : `array`
        An `(n_images, n_labels)` array with the one-hot labels.
    """
    # Get number of labels and classes
    num_labels = labels_dense.shape[0]
    num_classes = labels_dense.max() + 1
    # Create binary one-hot indicator
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, as_one_hot=False, verbose=False):
    r"""
    Extract labels from gzip file.

    Parameters
    ----------
    filename : `pathlib.PosixPath`
        The gzip file path.
    as_one_hot : `bool`, optional
        If `False`, then the labels are returned as integers within
        a `(n_images,)` numpy array. If `True`, then the labels are
        returned as one-hot vetors in an `(n_images, n_labels)` numpy
        array.
    verbose : `bool`, optional
        If `True`, then the progress will be printed.

    Returns
    -------
    labels : `array`
        The extracted labels.
    """
    if verbose:
        print_dynamic('Extracting {}'.format(filename))
    with open(str(filename), 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        # Convert labels array to one-hot labels, if required
        if as_one_hot:
            return _convert_dense_to_one_hot(labels)
        return labels


def split_data(images, labels, n_images):
    r"""
    Method that splits a set of images and corresponding labels in two disjoint
    sets. This is useful for creating a training and validation set.

    Parameters
    ----------
    images : `array` or `list`
        The images to split.
    labels : `array`
        The corresponding labels to split.
    n_images : `int`
        The number of images of the first disjoint set.

    Returns
    -------
    images1 : `array` or `list`
        The first set of images.
    labels1 : `array`
        The first set of labels.
    images2 : `array` or `list`
        The second set of images.
    labels2 : `array`
        The second set of labels.
    """
    images1 = images[:n_images]
    labels1 = labels[:n_images]
    images = images[n_images:]
    labels = labels[n_images:]
    return images1, labels1, images, labels


def convert_images_to_array(images):
    r"""
    Method that converts a list of `menpo.image.Image` objects to numpy array
    of shape `(n_images, height, width, n_channels)`.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The list of images.

    Returns
    -------
    images : `array`
        The `(n_images, height, width, n_channels)` array of images.
    """
    if isinstance(images, list):
        # Get dimensions
        n_images = len(images)
        height, width = images[0].shape
        n_channels = images[0].n_channels
        # Initialize array with zeros
        arr = np.zeros((n_images, height, width, n_channels),
                       dtype=images[0].pixels.dtype)
        # Extract pixels from each image
        for i, im in enumerate(images):
            arr[i] = im.pixels_with_channels_at_back()[..., None]
        return arr
    else:
        return images


def import_mnist_data(n_validation_images=5000, as_one_hot=False, verbose=False):
    r"""
    Method that downloads, extracts and converts to appropriate format the MNIST
    data. It returns the train, validation and test images with the corresponding
    labels.

    Parameters
    ----------
    n_validation_images : `int`, optional
        The number of images from the training set that will be used as
        validation set.
    as_one_hot : `bool`, optional
        If `False`, then the labels are returned as integers within
        a `(n_images,)` numpy array. If `True`, then the labels are
        returned as one-hot vectors in an `(n_images, n_labels)` numpy array.
    verbose : `bool`, optional
        If `True`, then the progress will be printed.

    Returns
    -------
    train_images : `list` of `menpo.image.Image`
        The list of train images.
    train_labels : `array`
        The array of labels of the train images.
    validation_images : `list` of `menpo.image.Image`
        The list of validation images.
    validation_labels : `array`
        The array of labels of the validation images.
    test_images : `list` of `menpo.image.Image`
        The list of test images.
    test_labels : `array`
        The array of labels of the test images.
    """
    # Download MNIST, if is not already downloaded
    train_images_path = download(TRAIN_IMAGES, verbose=verbose)
    train_labels_path = download(TRAIN_LABELS, verbose=verbose)
    test_images_path = download(TEST_IMAGES, verbose=verbose)
    test_labels_path = download(TEST_LABELS, verbose=verbose)

    # Extract the gz files and convert them to appropriate format
    train_images = extract_images(train_images_path, as_images=True,
                                  verbose=verbose)
    train_labels = extract_labels(train_labels_path, as_one_hot=as_one_hot,
                                  verbose=verbose)
    test_images = extract_images(test_images_path, as_images=True,
                                 verbose=verbose)
    test_labels = extract_labels(test_labels_path, as_one_hot=as_one_hot,
                                 verbose=verbose)

    # Generate a validation set from the training set
    validation_images, validation_labels, train_images, train_labels = \
        split_data(train_images, train_labels, n_validation_images)

    if verbose:
        print_dynamic('Successfully imported MNIST')

    # Return images and labels
    return (train_images, train_labels, validation_images, validation_labels,
            test_images, test_labels)


def _int64_feature(value):
    r"""
    Convenience method for defining a 64-bit integer within tensorflow.

    Parameters
    ----------
    value : `int`
        The input value.

    Returns
    -------
    tf_int64 : `tf.train.Feature`
        The converted value.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    r"""
    Convenience method for defining a bytes list within tensorflow.

    Parameters
    ----------
    value : `bytes`
        The input bytes list.

    Returns
    -------
    tf_bytes : `tf.train.Feature`
        The converted value.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_data(images, labels, filename, verbose=False):
    r"""
    Method that saves the provided images and labels to tfrecords file using the
    tensorflow record writer.

    Parameters
    ----------
    images : `list` or `array`
        The images to serialize.
    labels : `array`
        The corresponding labels.
    filename : `str`
        The filename to use. Note that the data will be saved in the 'data'
        folder.
    verbose : `bool`, optional
        If `True`, then the progress will be printed.
    """
    # If images is list, convert it to numpy array
    images = convert_images_to_array(images)

    # Get number of images, height, width and number of channels
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match labels size %d.".format(images.shape[0], num_examples))
    height = images.shape[1]
    width = images.shape[2]
    n_channels = images.shape[3]

    # Save data
    filename = str(src_dir_path() / 'data' / (filename + '.tfrecords'))
    if verbose:
        print_dynamic('Writing {}'.format(filename))
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(n_channels),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    if verbose:
        print_dynamic('Completed successfully!')


if __name__ == '__main__':
    # Import MNIST data
    (train_images, train_labels, validation_images, validation_labels,
     test_images, test_labels) = import_mnist_data(verbose=True)

    # Serialize MNIST data
    serialize_data(train_images, train_labels, 'train', verbose=True)
    serialize_data(validation_images, validation_labels, 'validation',
                   verbose=True)
    serialize_data(test_images, test_labels, 'test', verbose=True)
