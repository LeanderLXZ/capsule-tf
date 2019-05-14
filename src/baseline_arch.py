from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.layers import *
from models.capsule_layers import *
from models.loss_funcitons import *


def model_arch(inputs, labels, input_imgs,
               cfg, is_training=None, restore_vars_dict=None):

  if cfg.CLF_LOSS == 'margin':
    clf_loss_fn = margin_loss
    clf_loss_params = cfg.MARGIN_LOSS_PARAMS
  elif cfg.CLF_LOSS == 'margin_h':
    clf_loss_fn = margin_loss_h
    clf_loss_params = cfg.MARGIN_LOSS_H_PARAMS
  else:
    raise ValueError('Wrong CLF_LOSS Name!')

  model = Sequential(inputs, verbose=True)

  if restore_vars_dict is not None:

    with tf.variable_scope('classifier'):

      model.add(NHWC2NCHW())
      model.add(Conv(
          cfg,
          kernel_size=9,
          stride=1,
          n_kernel=256,
          padding='VALID',
          act_fn='relu',
          idx=0
      ))
      model.add(Capsule4Dto5D())
      model.add(ConvSlimCapsule(
          cfg,
          output_dim=32,
          output_atoms=8,
          num_routing=3,
          leaky=False,
          kernel_size=3,
          stride=2,
          padding='VALID',
          act_fn='squash',
          idx=0
      ))
      model.add(Capsule5Dto3D())
      model.add(Capsule(
          cfg,
          output_dim=10,
          output_atoms=16,
          num_routing=3,
          leaky=False,
          act_fn='squash',
          idx=1
      ))
      model.add_name('clf_logits')

      clf_loss, clf_preds = model.get_loss(
          clf_loss_fn, labels, **clf_loss_params)
      clf_loss = tf.identity(clf_loss, name='clf_loss')
      clf_preds = tf.identity(clf_preds, name='clf_preds')

    with tf.variable_scope('reconstruction'):

      model.add(Mask(labels))
      model.add(Dense(
          cfg,
          output_dim=512,
          act_fn='relu',
          idx=0))
      model.add(Dense(
          cfg,
          output_dim=1024,
          act_fn='relu',
          idx=1))
      model.add(Dense(
          cfg,
          output_dim=cfg.IMAGE_SIZE[0] * cfg.IMAGE_SIZE[1],
          act_fn=None,
          idx=2))
      model.add_name('rec_logits')

      rec_loss_params = {
        'decoder_type': 'fc',
        'rec_loss_type': 'ce'
      }
      rec_loss, rec_imgs = model.get_loss(
          reconstruction_loss, input_imgs, **rec_loss_params)
      rec_loss = tf.identity(rec_loss, name='rec_loss')
      rec_imgs = tf.identity(rec_imgs, name='rec_imgs')

    loss = clf_loss + cfg.REC_LOSS_SCALE * rec_loss
    loss = tf.identity(loss, name='loss')

  # Fine-tuning
  # ============================================================================
  else:

    w_conv_0 = restore_vars_dict['w_conv_0']
    b_conv_0 = restore_vars_dict['b_conv_0']
    w_caps_0 = restore_vars_dict['w_caps_0']
    b_caps_0 = restore_vars_dict['b_caps_0']
    w_caps_1 = restore_vars_dict['w_caps_1']
    b_caps_1 = restore_vars_dict['b_caps_1']

    with tf.variable_scope('classifier'):

      model.add(NHWC2NCHW())
      model.add(Conv(
          cfg,
          kernel_size=9,
          stride=1,
          n_kernel=256,
          padding='VALID',
          act_fn='relu',
          idx=0
      ).assign_variables(w_conv_0, b_conv_0))
      model.add(Capsule4Dto5D())
      model.add(ConvSlimCapsule(
          cfg,
          output_dim=32,
          output_atoms=8,
          num_routing=3,
          leaky=False,
          kernel_size=3,
          stride=2,
          padding='VALID',
          act_fn='squash',
          idx=0
      ).assign_variables(w_caps_0, b_caps_0))
      model.add(Capsule5Dto3D())
      model.add(Capsule(
          cfg,
          output_dim=10,
          output_atoms=16,
          num_routing=3,
          leaky=False,
          act_fn='squash',
          idx=1
      ).assign_variables(w_caps_1, b_caps_1))
      model.add(Capsule(
          cfg,
          output_dim=90,
          output_atoms=16,
          num_routing=3,
          leaky=False,
          act_fn='squash',
          idx=1
      ))
      model.add_name('clf_logits')

      clf_loss, clf_preds = model.get_loss(
          clf_loss_fn, labels, **clf_loss_params)
      clf_loss = tf.identity(clf_loss, name='clf_loss')
      clf_preds = tf.identity(clf_preds, name='clf_preds')

    with tf.variable_scope('reconstruction'):

      model.add(Mask(labels))
      model.add(Dense(
          cfg,
          output_dim=512,
          act_fn='relu',
          idx=0))
      model.add(Dense(
          cfg,
          output_dim=1024,
          act_fn='relu',
          idx=1))
      model.add(Dense(
          cfg,
          output_dim=cfg.IMAGE_SIZE[0] * cfg.IMAGE_SIZE[1],
          act_fn=None,
          idx=2))
      model.add_name('rec_logits')

      rec_loss_params = {
        'decoder_type': 'fc',
        'rec_loss_type': 'ce'
      }
      rec_loss, rec_imgs = model.get_loss(
          reconstruction_loss, input_imgs, **rec_loss_params)
      rec_loss = tf.identity(rec_loss, name='rec_loss')
      rec_imgs = tf.identity(rec_imgs, name='rec_imgs')

    loss = clf_loss + cfg.REC_LOSS_SCALE * rec_loss
    loss = tf.identity(loss, name='loss')

  return loss, clf_loss, clf_preds, rec_loss, rec_imgs, model.info
