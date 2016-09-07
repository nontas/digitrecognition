# MNIST Digit Recognition
This package provides training and evaluation code to perform Digit Recognition 
with Convolutional Neural Network on the 
[MNIST dataset](http://yann.lecun.com/exdb/mnist/).

1. [Installation](#installation)
2. [Methodology](#methodology)
3. [Running](#running)
4. [Results](#results)

-------------------------------

## 1. Installation
The package depends on:
 
1. [**TF-Slim**](https://github.com/tensorflow/models/blob/master/inception/inception/slim/README.md): 
The lightweight syntactic sugar library of [TensorFlow](https://www.tensorflow.org/)
2. [**Menpo Project**](http://www.menpo.org/): The Python framework for data handling and deformable modeling.

In general, as explained in [Menpo's installation instructions](http://www.menpo.org/installation/), 
it is highly recommended to use [conda](http://conda.pydata.org/miniconda.html) as your Python distribution.

Once downloading and installing [conda](http://conda.pydata.org/miniconda.html), this project can be installed by:

**Step 1:** Create a new conda environment and activate it:
```console
$ conda create -n mnist python=3.5
$ source activate mnist
```

**Step 2:** Install [TensorFlow](https://www.tensorflow.org/) following the 
official [installation instructions](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html). 
For example, for 64-bit Linux, the installation of GPU enabled, Python 3.5 TensorFlow involves:
```console
(mnist)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl
(mnist)$ pip install --upgrade $TF_BINARY_URL
```

**Step 3:** Install [menpo](https://github.com/menpo/menpo) from the _menpo_ channel as:
```console
(mnist)$ conda install -c menpo menpo
```

**Step 4:** Clone and install the `digitrecognition` project as:
```console
(mnist)$ cd ~/Documents
(mnist)$ git clone git@github.com:nontas/digitrecognition.git
(mnist)$ pip install -e digitrecognition/
```

-------------------------------

## 2. Methodology
The solution implemented in this package is the following:

* _Data pre-processing_  
  During training, each image is pre-processed in the following way:
  * The image is rotated around its centre with a random angle.
  * The image is skewed (distorted) with random angles.
  
  The pre-proceessing is happenning on the fly, i.e. every time a new example is 
  loaded the pre-processing is applied. Given that we also do not limit the number of batches, 
  the system will keep training with an infinite number of randomly perturbed examples.
  This pre-processing is implemented using [Menpo](https://github.com/menpo/menpo). 
  You can find the implementation in [data_provider.py](https://github.com/nontas/digitrecognition/blob/master/digitrecognition/data_provider.py).
  To get an idea of the results of the employed pre-processing, you 
  can run the following code in a [Jupyter notebook](https://github.com/jupyter/notebook):
  ```python
  %matplotlib inline
  from numpy.random import randint
  import matplotlib.pyplot as plt
  from menpo.image import Image
  
  from digitrecognition import import_mnist_data
  from digitrecognition.data_provider import preprocess
  
  # Load train images
  train_images, _, _, _, _, _ = import_mnist_data(verbose=True)
  
  # Generate random image index
  i = randint(0, len(train_images))
  
  # Pre-process image
  im = train_images[i].pixels_with_channels_at_back()[..., None]
  im = preprocess(im)
  
  # Plot before and after
  plt.subplot(121)
  train_images[i].view()
  plt.subplot(122)
  Image.init_from_channels_at_back(im).view()
  ```
  
* _Network architecture:_  
  After trying a few network architectures, the best performing one is:
  1. Convolutional layer (64 filters, 5x5 kernel, batch normalization)
  2. Max-Pooling layer (2x2 kernel)
  3. Convolutional layer (32 filters, 5x5 kernel, batch normalization)
  4. Max-Pooling layer (2x2 kernel)
  5. Fully Connected layer (1024 outputs, batch normalization)
  6. Fully Connected layer (10 outputs)
  
  The definitions of the various architectures can be found in
  [model.py](https://github.com/nontas/digitrecognition/blob/master/digitrecognition/model.py).
  
* _Learning rate decay:_  
  The experiments showed that to decay the learning rate can help a lot. The initial 
  value of the learning rate is `0.001` and then it decreases with a rate of `0.9` 
  every `10000` steps. Refer to [train.py](https://github.com/nontas/digitrecognition/blob/master/digitrecognition/train.py) for more
  details on how this is implemented.
  
* _Optimizer:_  
  The employed optimizer is `tf.train.AdamOptimizer` which proved better than 
  `tf.train.RMSPropOptimizer`.

-------------------------------

## 3. Running
To run the training and evaluation, do the following:

**Data Collection:** In the terminal, run 
```console
(mnist)$ python digitrecognition/data_converter.py
```
which will download the [MNIST](http://yann.lecun.com/exdb/mnist/) data, if are not 
already downloaded, load them and convert them to `tfrecords` files. 
The files are stored in the `data/` folder. Note that the data will be split in the
training, validation and testing sets.

**Training:** To train the model, run:
```console
(mnist)$ python digitrecognition/train.py
```
which will initiate the training. Various arguments can be passed in through that 
function:
```console
  --architecture ARCHITECTURE
                        The network architecture to use: baseline, ultimate,
                        ultimate_v2.
  --batch_size BATCH_SIZE
                        Batch size.
  --num_train_batches NUM_TRAIN_BATCHES
                        Number of batches to train (epochs).
  --log_train_dir LOG_TRAIN_DIR
                        Directory with the training log data.
  --initial_learning_rate INITIAL_LEARNING_RATE
                        Initial value of learning rate decay.
  --decay_steps DECAY_STEPS
                        Learning rate decay steps.
  --decay_rate DECAY_RATE
                        Learning rate decay rate.
  --eval_set EVAL_SET   The dataset to evaluate on: train, validation or test
  --num_samples NUM_SAMPLES
                        Number of samples to evaluate.
  --log_eval_dir LOG_EVAL_DIR
                        Directory with the evaluation log data.
  --momentum MOMENTUM   Optimizer .
  --optimization OPTIMIZATION
                        The optimization method to use. Either 'rms' or
                        'adam'.
  --verbose [VERBOSE]   Print log in terminal.
```
These arguments and their default values are defined in [_params.py_](https://github.com/nontas/digitrecognition/blob/master/digitrecognition/params.py).
Note that by default, the training log files are stored in `./log/train/`.

**Evaluation:** To evaluate the model on the validation set, run:
```console
(mnist)$ python digitrecognition/eval.py
```
Note that by default, the validation log files are stored in `./log/eval/`.

**Testing:** Testing can be performed as:
```console
(mnist)$ python digitrecognition/eval.py --eval_set=test --log_eval_dir=./log/eval_test
```

**TensorBoard:** You can simultaneously run the training and validation. The results can 
be observed through [TensorBoard](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html). 
Simply run:
```console
(mnist)$ tensorboard --logdir=log
```
This makes it easy to explore the graph, data, loss evolution and accuracy on the validation set. 
 
-------------------------------
 
## 4. Results
The results of the above model are:
