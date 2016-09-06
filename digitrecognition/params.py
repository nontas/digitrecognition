import tensorflow as tf

from digitrecognition.base import src_dir_path

# Initialize the argparse wrapper
flags = tf.app.flags

# Define parameters related to training
flags.DEFINE_string('architecture', 'ultimate',
                    'The network architecture to use: baseline, ultimate, '
                    'ultimate_v2.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_train_batches', None,
                     'Number of batches to train (epochs).')
flags.DEFINE_string('log_train_dir', str(src_dir_path() / 'log' / 'train'),
                    'Directory with the training log data.')

# Learning rate decay parameters
flags.DEFINE_float('initial_learning_rate', 0.001,
                   'Initial value of learning rate decay.')
flags.DEFINE_float('decay_steps', 10000, 'Learning rate decay steps.')
flags.DEFINE_float('decay_rate', 0.9, 'Learning rate decay rate.')

# Define parameters related to evaluation (validation)
flags.DEFINE_string('eval_set', 'validation',
                    'The dataset to evaluate on: train, validation or test')
flags.DEFINE_integer('num_samples', 5000, 'Number of samples to evaluate.')
flags.DEFINE_string('log_eval_dir', str(src_dir_path() / 'log' / 'eval'),
                    'Directory with the evaluation log data.')

# Optimizer parameters
flags.DEFINE_float('momentum', 0.9, 'Optimizer .')
flags.DEFINE_string('optimization', 'adam',
                    "The optimization method to use. Either 'rms' or 'adam'.")

# Verbose flag
flags.DEFINE_bool('verbose', True, 'Print log in terminal.')
