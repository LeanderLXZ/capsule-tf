from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def squash(x, epsilon=1e-9):
  """Applies norm non-linearity (squash) to a capsule layer.

  Args:
    x: A tensor with shape: (batch_size, num_caps, vec_dim[, 1]).
    epsilon: Add epsilon(a very small number) to zeros

  Returns:
    A tensor with the same shape as input tensor but squashed in 'vec_dim'
    dimension.
  """
  if len(x.get_shape().as_list()) == 4:
    vec_squared_norm = tf.reduce_sum(tf.square(x), -2, keep_dims=True)
  else:
    vec_squared_norm = tf.reduce_sum(tf.square(x), -1, keep_dims=True)
  scalar_factor = tf.div(vec_squared_norm, 1 + vec_squared_norm)
  unit_vec = tf.div(x, tf.sqrt(vec_squared_norm + epsilon))
  return tf.multiply(scalar_factor, unit_vec)


def squash_v2(x):
  """Squashing function version 2.0

  Args:
    x: Input tensor. Shape is [batch, num_channels, num_atoms] for
      a fully connected capsule layer or
      [batch, num_channels, num_atoms, height, width] for a convolutional
      capsule layer.

  Returns:
    A tensor with same shape as input (rank 3) for output of this layer.
  """
  with tf.name_scope('norm_non_linearity'):
    norm = tf.norm(x, axis=2, keep_dims=True)
    norm_squared = norm * norm
    return (x / norm) * (norm_squared / (1 + norm_squared))

