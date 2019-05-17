from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import utils


def margin_loss(logits,
                labels,
                m_plus=0.9,
                m_minus=0.1,
                lambda_=0.5):
  """Calculate margin loss according to Hinton's paper.

  L = T_c * max(0, m_plus-||v_c||)^2 +
      lambda_ * (1-T_c) * max(0, ||v_c||-m_minus)^2

  Args:
    logits: output tensor of capsule layers.
      - shape: (batch_size, num_caps, vec_dim)
    labels: labels
      - shape: (batch_size, num_caps)
    m_plus: truncation of positive item.
    m_minus: truncation of negative item.
    lambda_: lambda.

  Returns:
    loss: margin loss.
    preds: predicted tensor.
  """
  preds = utils.get_vec_length(logits)
  max_square_plus = tf.square(tf.maximum(0., m_plus - preds))
  max_square_minus = tf.square(tf.maximum(0., preds - m_minus))
  # max_square_plus & max_plus shape: (batch_size, num_caps)

  loss_c = tf.multiply(labels, max_square_plus) + \
      lambda_ * tf.multiply((1 - labels), max_square_minus)

  # Total margin loss
  loss = tf.reduce_mean(tf.reduce_sum(loss_c, axis=1))

  return loss, preds


def margin_loss_h(logits,
                  labels,
                  margin=0.4,
                  down_weight=0.5):
  """Penalizes deviations from margin for each logit.

  Each wrong logit costs its distance to margin. For negative logits margin is
  0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
  margin is 0.4 from each side.

  Args:
    logits: tensor, model predictions vectors.
    labels: tensor, one hot encoding of ground truth.
    margin: scalar, the margin after subtracting 0.5 from raw_logits.
    down_weight: scalar, the factor for negative cost.

  Returns:
    loss: A scalar with cost for all data point.
    preds: predicted tensor.
  """
  preds = utils.get_vec_length(logits, 1e-9)
  logits = preds - 0.5
  positive_cost = labels * tf.cast(tf.less(logits, margin),
                                   tf.float32) * tf.pow(logits - margin, 2)
  negative_cost = (1 - labels) * tf.cast(
      tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
  loss_c = 0.5 * positive_cost + down_weight * 0.5 * negative_cost
  loss = tf.reduce_mean(tf.reduce_sum(loss_c, axis=1))
  return loss, preds


def reconstruction_loss(logits,
                        input_imgs,
                        decoder_type='fc',
                        rec_loss_type='ce'):
  """Calculate loss with reconstruction.

  Args:
    logits: output tensor of models
      - shape (batch_size, num_caps, vec_dim)
    input_imgs: ground truth input images
      - shape (batch_size, *image_size)
    decoder_type: 'fc' or 'conv' or 'conv_t'
    rec_loss_type: 'mse' or 'ce'

  Return:
    loss: reconstruction loss.
    rec_imgs: reconstructed images.
  """
  if rec_loss_type == 'mse':
    inputs_flatten = tf.contrib.layers.flatten(input_imgs)
    if decoder_type != 'fc':
      reconstructed = tf.contrib.layers.flatten(logits)
    else:
      reconstructed = logits
    loss = tf.reduce_mean(
        tf.square(reconstructed - inputs_flatten))
    rec_imgs = logits

  elif rec_loss_type == 'ce':
    if decoder_type == 'fc':
      inputs_imgs_ = tf.contrib.layers.flatten(input_imgs)
    else:
      inputs_imgs_ = input_imgs
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=inputs_imgs_, logits=logits))
    rec_imgs = tf.nn.sigmoid(logits)

  else:
    raise ValueError('Wrong reconstruction loss type!')

  imgs_shape = input_imgs.get_shape().as_list()
  rec_imgs = tf.reshape(
      rec_imgs, shape=[-1, *imgs_shape[1:]], name='rec_images')

  return loss, rec_imgs
