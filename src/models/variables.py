from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def tf_variable(name,
                shape,
                initializer,
                dtype=tf.float32,
                store_on_cpu=True,
                trainable=True,
                verbose=False):
  """
  Helper to create a normal Variable.

  Args:
    name: name of the tf_variable
    shape: list of ints
    initializer: initializer for Variable
    dtype: data type
    store_on_cpu: create a Variable stored on CPU memory
    trainable: tf_variable can be trained by models
    verbose: if set add histograms.

  Returns:
    Variable Tensor
  """
  if store_on_cpu:
    with tf.device('/cpu:0'):
      with tf.name_scope(name):
        var = tf.get_variable(name, shape, initializer=initializer,
                              dtype=dtype, trainable=trainable)
  else:
    with tf.name_scope(name):
      var = tf.get_variable(name, shape, initializer=initializer,
                            dtype=dtype, trainable=trainable)

  variable_summaries(var, verbose)
  return var


def variable_summaries(var, verbose):
  """Attaches a lot of summaries to a Tensor (for TensorBoard visualization).

  Args:
    var: tensor, statistic summaries of this tensor is added.
    verbose: if set add histograms.
  """
  if verbose:
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
  else:
    pass


def activation_summary(x, verbose):
  """Creates summaries for activations.

  Creates a summary that provides a histogram and sparsity of activations.

  Args:
    x: Tensor
    verbose: if set add histograms.
  """
  if verbose:
    tf.summary.histogram('activations', x)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))
  else:
    pass
