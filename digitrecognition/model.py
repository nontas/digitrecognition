import tensorflow.contrib.slim as slim


def baseline(images):
    r"""
    Define a baseline architecture. It consists of:

        1. Convolutional layer (32 filters, 5x5 kernel)
        2. Max-Pooling layer (2x2 kernel)
        3. Convolutional layer (64 filters, 5x5 kernel)
        4. Max-Pooling layer (2x2 kernel)
        5. Fully Connected layer (500 outputs)
        6. Fully Connected layer (10 outputs)

    Parameters
    ----------
    images : `tf.Tensor`
        The float tensor with shape `(n_images, height, width, n_channels)`.

    Returns
    -------
    net : `tf.Tensor`
        The output of the network.
    """
    net = slim.layers.conv2d(images, 32, 5, scope='conv1')
    net = slim.layers.max_pool2d(net, 2, scope='pool1')
    net = slim.layers.conv2d(net, 64, 5, scope='conv2')
    net = slim.layers.max_pool2d(net, 2, scope='pool2')
    net = slim.layers.flatten(net, scope='flatten3')
    net = slim.layers.fully_connected(net, 500, scope='fc4')
    net = slim.layers.fully_connected(net, 10, activation_fn=None, scope='fc5')

    return net


def ultimate(images):
    r"""
    Define a network architecture. It consists of:

        1. Convolutional layer (64 filters, 5x5 kernel, batch normalization)
        2. Max-Pooling layer (2x2 kernel)
        3. Convolutional layer (32 filters, 5x5 kernel, batch normalization)
        4. Max-Pooling layer (2x2 kernel)
        5. Fully Connected layer (1024 outputs, batch normalization)
        6. Fully Connected layer (10 outputs)

    Parameters
    ----------
    images : `tf.Tensor`
        The float tensor with shape `(n_images, height, width, n_channels)`.

    Returns
    -------
    net : `tf.Tensor`
        The output of the network.
    """
    with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                        normalizer_fn=slim.batch_norm):
        net = slim.layers.conv2d(images, 64, 5, scope='conv1')
        net = slim.layers.max_pool2d(net, 2, scope='pool1')
        net = slim.layers.conv2d(net, 32, 5, scope='conv2')
        net = slim.layers.max_pool2d(net, 2, scope='pool2')
        net = slim.layers.flatten(net, scope='flatten3')
        net = slim.layers.dropout(net, scope='dropout4')
        net = slim.layers.fully_connected(net, 1024, scope='fc5')
    net = slim.layers.fully_connected(net, 10, activation_fn=None, scope='fc6')

    return net


def ultimate_v2(images):
    r"""
    Define a network architecture (similar to ultimate). It consists of:

        1. Convolutional layer (32 filters, 5x5 kernel, batch normalization)
        2. Max-Pooling layer (2x2 kernel)
        3. Convolutional layer (32 filters, 5x5 kernel, batch normalization)
        4. Max-Pooling layer (2x2 kernel)
        5. Fully Connected layer (512 outputs)
        6. Fully Connected layer (10 outputs)

    Parameters
    ----------
    images : `tf.Tensor`
        The float tensor with shape `(n_images, height, width, n_channels)`.

    Returns
    -------
    net : `tf.Tensor`
        The output of the network.
    """
    with slim.arg_scope([slim.layers.conv2d], normalizer_fn=slim.batch_norm):
        net = slim.layers.conv2d(images, 32, 5, scope='conv1')
        net = slim.layers.max_pool2d(net, 2, scope='pool1')
        net = slim.layers.conv2d(net, 32, 5, scope='conv2')
        net = slim.layers.max_pool2d(net, 2, scope='pool2')
        net = slim.layers.flatten(net, scope='flatten3')
        net = slim.layers.dropout(net, scope='dropout4')
        net = slim.layers.fully_connected(net, 512, scope='fc5')
    net = slim.layers.fully_connected(net, 10, activation_fn=None, scope='fc6')

    return net
