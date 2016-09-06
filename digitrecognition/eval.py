import os
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist

from digitrecognition.train import inputs, lenet
import digitrecognition.params

FLAGS = tf.app.flags.FLAGS


def evaluate(batch_size, num_batches, log_dir, checkpoint_dir):
    # Delete log
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    # Define network
    images, labels = inputs(train=False, batch_size=batch_size, num_epochs=None)
    predictions = lenet(images)

    # Convert predictions to numbers
    predictions = tf.to_int32(tf.argmax(predictions, 1))

    # Report the streaming accuracy (running accuracy)
    tf.scalar_summary('accuracy', slim.metrics.accuracy(predictions, labels))
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map(
        {"streaming_mse": slim.metrics.streaming_mean_squared_error(predictions,
                                                                    labels)})

    # Define the streaming summaries for tensorboard
    for metric_name, metric_value in metrics_to_values.items():
        tf.scalar_summary(metric_name, metric_value)

    # Evaluate every 30 seconds
    slim.evaluation.evaluation_loop(
        '', checkpoint_dir, log_dir, num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        summary_op=tf.merge_all_summaries(),
        eval_interval_secs=60)


def main(_):
    evaluate(FLAGS.batch_size,
             FLAGS.num_eval_batches,
             FLAGS.log_eval_dir,
             FLAGS.log_train_dir)


if __name__ == '__main__':
    tf.app.run()
