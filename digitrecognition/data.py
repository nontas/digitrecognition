import os
from urllib.request import urlretrieve
import gzip
import numpy as np

from menpo.image import Image
from menpo.visualize import print_dynamic

from .base import src_dir_path


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
        urlretrieve(SOURCE_URL + filename, filename=str(file_path))
    # Return
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
    Extract images from gz file.

    Parameters
    ----------
    filename : `pathlib.PosixPath`
        The gz file path.
    as_images : `bool`, optional
        If `True`, then the method returns a list containing a
        `menpo.image.Image` per image. If `False`, then it
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
        if as_images:
            return [Image(data[i, :, :, 0]) for i in range(data.shape[0])]
        return data


def dense_to_one_hot(labels_dense):
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
    Extract labels from gz file.

    Parameters
    ----------
    filename : `pathlib.PosixPath`
        The gz file path.
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
        The labels.
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
        if as_one_hot:
            return dense_to_one_hot(labels)
        return labels


def split_data(images, labels, n_images):
    r"""
    Method that splits a set of images and corresponding
    labels in two disjoint sets.

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
    Method that converts a list of image objects to numpy array
    of shape `(n_images, height, width, n_channels)`.

    Parameters
    ----------
    images : `list`
        The list of images.

    Returns
    -------
    images : `array`
        The `(n_images, height, width, n_channels)` array of images.
    """
    if isinstance(images, list):
        n_images = len(images)
        height, width = images[0].shape
        n_channels = images[0].n_channels
        arr = np.zeros((n_images, height, width, n_channels),
                       dtype=images[0].pixels.dtype)
        for i, im in enumerate(images):
            arr[i] = im.pixels_with_channels_at_back()[..., None]
        return arr
    return images


def import_mnist_data(n_validation_images=5000, as_one_hot=False, verbose=False):
    r"""
    Method that downloads, extracts and converts to appropriate format the MNIST
    data. It returns the train, validation and test images with the corresponding
    labels.

    Parameters
    ----------
    n_validation_images : `int`, optional
        The number of validation images.
    as_one_hot : `bool`, optional
        If `False`, then the labels are returned as integers within
        a `(n_images,)` numpy array. If `True`, then the labels are
        returned as one-hot vetors in an `(n_images, n_labels)` numpy
        array.
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

    # Augment training data

    if verbose:
        print_dynamic('Successfully imported MNIST')

    return (train_images, train_labels, validation_images, validation_labels,
            test_images, test_labels)
