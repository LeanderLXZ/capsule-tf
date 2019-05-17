from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from models import utils


class Model(object):

  def __init__(self,
               cfg,
               model_arch,
               restore_vars_dict=None):

    self.cfg = cfg
    self.batch_size = cfg.BATCH_SIZE
    self.model_arch = model_arch
    self.restore_vars_dict = restore_vars_dict
    self.model_arch_info = None

  def _get_inputs(self,
                  input_size,
                  num_class,
                  image_size):
    """Get input tensors.

    Args:
      input_size: the size of input tensor
      num_class: number of class of label
      image_size: the size of ground truth images, should be 3 dimensional
    Returns:
      input tensors
    """
    _inputs = tf.placeholder(
        tf.float32, shape=[self.cfg.BATCH_SIZE, *input_size], name='inputs')
    _labels = tf.placeholder(
        tf.float32, shape=[self.cfg.BATCH_SIZE, num_class], name='labels')
    _input_imgs = tf.placeholder(
        tf.float32, shape=[self.cfg.BATCH_SIZE, *image_size], name='input_imgs')
    _is_training = tf.placeholder(tf.bool, name='is_training')

    return _inputs, _labels, _input_imgs, _is_training

  def _optimizer(self,
                 opt_name='adam',
                 n_train_samples=None,
                 global_step=None):
    """Optimizer."""
    # Learning rate with exponential decay
    if self.cfg.LR_DECAY:
      learning_rate_ = tf.train.exponential_decay(
          learning_rate=self.cfg.LEARNING_RATE,
          global_step=global_step,
          decay_steps=self.cfg.LR_DECAY_STEPS,
          decay_rate=self.cfg.LR_DECAY_RATE)
      learning_rate_ = tf.maximum(learning_rate_, 1e-6)
    else:
      learning_rate_ = self.cfg.LEARNING_RATE

    if opt_name == 'adam':
      return tf.train.AdamOptimizer(learning_rate_)

    elif opt_name == 'momentum':
      n_batches_per_epoch = \
          n_train_samples // self.cfg.GPU_BATCH_SIZE * self.cfg.GPU_NUMBER
      boundaries = [
          n_batches_per_epoch * x
          for x in np.array(self.cfg.LR_BOUNDARIES, dtype=np.int64)]
      staged_lr = [self.cfg.LEARNING_RATE * x
                   for x in self.cfg.LR_STAGE]
      learning_rate = tf.train.piecewise_constant(
          global_step,
          boundaries, staged_lr)
      return tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=self.cfg.MOMENTUM)

    elif opt_name == 'gd':
      return tf.train.GradientDescentOptimizer(learning_rate_)

    else:
      raise ValueError('Wrong optimizer name!')

  def _inference(self,
                 inputs,
                 labels,
                 input_imgs,
                 num_class,
                 is_training=None):
    """Build inference graph.

    Args:
      inputs: input tensor.
        - shape (batch_size, *input_size)
      labels: labels tensor.
      num_class: number of class of label.
      input_imgs: ground truth images.
      is_training: Whether or not the model is in training mode.

    Return:
      logits: output tensor of models
        - shape: (batch_size, num_caps, vec_dim)
    """
    loss, clf_loss, clf_preds, rec_loss, rec_imgs, self.model_arch_info = \
        self.model_arch(self.cfg, inputs, labels, input_imgs,
                        num_class=num_class, is_training=is_training,
                        restore_vars_dict=self.restore_vars_dict)

    # Accuracy
    correct_pred = tf.equal(
        tf.argmax(clf_preds, axis=1), tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(
        correct_pred, tf.float32), name='accuracy')

    return loss, accuracy, clf_loss, clf_preds, rec_loss, rec_imgs

  def build_graph(self,
                  input_size=(None, None, None),
                  image_size=(None, None, None),
                  num_class=None,
                  n_train_samples=None):
    """Build the graph of CapsNet.

    Args:
      input_size: size of input tensor
      image_size: the size of ground truth images, should be 3 dimensional
      num_class: number of class of label
      n_train_samples: number of train samples

    Returns:
      tuple of (global_step, train_graph, inputs, labels, train_op,
                saver, summary_op, loss, accuracy, classifier_loss,
                reconstruct_loss, reconstructed_images)
    """
    tf.reset_default_graph()
    train_graph = tf.Graph()

    with train_graph.as_default():

      # Get input placeholders
      inputs, labels, input_imgs, is_training = \
          self._get_inputs(input_size, num_class, image_size=image_size)

      # Global step
      global_step = tf.placeholder(tf.int16, name='global_step')

      # Optimizer
      optimizer = self._optimizer(global_step=global_step,
                                  opt_name=self.cfg.OPTIMIZER,
                                  n_train_samples=n_train_samples)

      # Build inference Graph
      loss, accuracy, clf_loss, clf_preds, rec_loss, rec_imgs = self._inference(
          inputs, labels, input_imgs, num_class=num_class, is_training=is_training)

      # Optimizer
      train_op = optimizer.minimize(loss)

      # Create a saver.
      saver = tf.train.Saver(tf.global_variables(),
                             max_to_keep=self.cfg.MAX_TO_KEEP_CKP)

      # Build the summary operation from the last tower summaries.
      tf.summary.scalar('accuracy', accuracy)
      tf.summary.scalar('loss', loss)
      if self.cfg.WITH_REC:
        tf.summary.scalar('clf_loss', clf_loss)
        tf.summary.scalar('rec_loss', rec_loss)
      summary_op = tf.summary.merge_all()

      return global_step, train_graph, inputs, labels, input_imgs, \
          is_training, train_op, saver, summary_op, loss, accuracy, \
          clf_loss, clf_preds, rec_loss, rec_imgs


class ModelDistribute(Model):

  def __init__(self,
               cfg,
               model_arch,
               restore_vars_dict=None):
    super(ModelDistribute, self).__init__(cfg, model_arch, restore_vars_dict)
    self.batch_size = cfg.BATCH_SIZE // cfg.GPU_NUMBER

  @staticmethod
  def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    This function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
                   is over individual gradients. The inner list is over the
                   gradient calculation for each tower.
        - shape: [[(grad0_gpu0, var0_gpu0), ..., (gradM_gpu0, varM_gpu0)],
                   ...,
                  [(grad0_gpuN, var0_gpuN), ..., (gradM_gpuN, varM_gpuN)]]

    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Each grad_and_vars looks like:
      # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for grad, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_grad = tf.expand_dims(grad, 0)
        # Append on a 'tower' dimension which we will average over.
        grads.append(expanded_grad)

      # grads: [[grad0_gpu0], [grad0_gpu1], ..., [grad0_gpuN]]
      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # The Variables are redundant because they are shared across towers.
      # So we will just return the first tower's pointer to the Variable.
      v = grad_and_vars[0][1]  # varI_gpu0
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)

    # average_grads: [(grad0, var0), (grad1, var1), ..., (gradM, varM)]
    return average_grads

  def _average_metrics(self, loss_all, acc_all, clf_loss_all,
                       clf_preds_all, rec_loss_all, rec_imgs_all):
    """Calculate average of metrics.

    Args:
      loss_all: final losses of each tower, list
      acc_all: accuracies of each tower, list
      clf_loss_all: classifier losses of each tower, list
      clf_preds_all: predictions of each tower, list
      rec_loss_all: reconstruction losses of each tower, list
      rec_imgs_all: reconstructed images of each tower, list of 4D tensor

    Returns:
      tuple of metrics
    """
    n_tower = float(len(loss_all))

    loss = tf.divide(
        tf.add_n(loss_all), n_tower, name='total_loss')
    accuracy = tf.divide(
        tf.add_n(acc_all), n_tower, name='total_acc')
    clf_preds = tf.concat(clf_preds_all, axis=0, name='total_clf_preds')

    if self.cfg.WITH_REC:
      clf_loss = tf.divide(
          tf.add_n(clf_loss_all), n_tower, name='total_clf_loss')
      rec_loss = tf.divide(
          tf.add_n(rec_loss_all), n_tower, name='total_rec_loss')
      rec_imgs = tf.concat(rec_imgs_all, axis=0, name='total_rec_imgs')
    else:
      clf_loss, rec_loss, rec_imgs = None, None, None

    return loss, accuracy, clf_loss, clf_preds, rec_loss, rec_imgs

  def _calc_on_gpu(self, gpu_idx, x_tower, y_tower,
                   imgs_tower, num_class, is_training, optimizer):

    # Calculate the loss for one tower.
    loss_tower, acc_tower, clf_loss_tower, clf_preds_tower, \
        rec_loss_tower, rec_imgs_tower = self._inference(
            x_tower, y_tower, imgs_tower, 
            num_class=num_class, is_training=is_training)

    # Calculate the gradients on this tower.
    grads_tower = optimizer.compute_gradients(loss_tower)

    return grads_tower, loss_tower, acc_tower, clf_loss_tower, \
        clf_preds_tower, rec_loss_tower, rec_imgs_tower

  def build_graph(self,
                  input_size=(None, None, None),
                  image_size=(None, None, None),
                  num_class=None,
                  n_train_samples=None):
    """Build the graph of CapsNet.

    Args:
      input_size: size of input tensor
      image_size: the size of ground truth images, should be 3 dimensional
      num_class: number of class of label
      n_train_samples: number of train samples

    Returns:
      tuple of (global_step, train_graph, inputs, labels, train_op,
                saver, summary_op, loss, accuracy, classifier_loss,
                reconstruct_loss, reconstructed_images)
    """
    tf.reset_default_graph()
    train_graph = tf.Graph()

    with train_graph.as_default(), tf.device('/cpu:0'):

      # Get inputs tensor
      inputs, labels, input_imgs, is_training = \
        self._get_inputs(input_size, num_class, image_size=image_size)

      # Global step
      global_step = tf.placeholder(tf.int16, name='global_step')

      # Optimizer
      optimizer = self._optimizer(opt_name=self.cfg.OPTIMIZER,
                                  n_train_samples=n_train_samples,
                                  global_step=global_step)

      # Split data for each tower
      x_splits_tower = tf.split(
          axis=0, num_or_size_splits=self.cfg.GPU_NUMBER, value=inputs)
      y_splits_tower = tf.split(
          axis=0, num_or_size_splits=self.cfg.GPU_NUMBER, value=labels)
      imgs_splits_tower = tf.split(
          axis=0, num_or_size_splits=self.cfg.GPU_NUMBER, value=input_imgs)

      # Calculate the gradients for each models tower.
      grads_all, loss_all, acc_all, clf_loss_all, clf_preds_all, \
          rec_loss_all, rec_imgs_all = [], [], [], [], [], [], []
      for i in range(self.cfg.GPU_NUMBER):
        utils.thin_line()
        print('Building tower: ', i)

        # Dequeues one batch for the GPU
        x_tower, y_tower, imgs_tower = \
            x_splits_tower[i], y_splits_tower[i], imgs_splits_tower[i]

        with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
              grads_tower, loss_tower, acc_tower, clf_loss_tower, \
                  clf_preds_tower, rec_loss_tower, rec_imgs_tower = \
                  self._calc_on_gpu(i, x_tower, y_tower, imgs_tower,
                                    num_class, is_training, optimizer)

              # Keep track of the gradients across all towers.
              grads_all.append(grads_tower)

              # Collect metrics of each tower
              loss_all.append(loss_tower)
              acc_all.append(acc_tower)
              clf_loss_all.append(clf_loss_tower)
              clf_preds_all.append(clf_preds_tower)
              rec_loss_all.append(rec_loss_tower)
              rec_imgs_all.append(rec_imgs_tower)

      # Calculate the mean of each gradient.
      grads = self._average_gradients(grads_all)

      # Calculate means of metrics
      loss, accuracy, clf_loss, clf_preds, rec_loss, rec_imgs = \
          self._average_metrics(loss_all, acc_all, clf_loss_all,
                                clf_preds_all, rec_loss_all, rec_imgs_all)

      # Show variables
      utils.thick_line()
      print('Variables: ')
      for v in tf.global_variables():
        print(v)

      # Apply the gradients to adjust the shared variables.
      apply_gradient_op = optimizer.apply_gradients(grads)

      # Track the moving averages of all trainable variables.
      if self.cfg.MOVING_AVERAGE_DECAY:
        variable_averages = tf.train.ExponentialMovingAverage(
            self.cfg.MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)
      else:
        train_op = apply_gradient_op

      # Create a saver.
      saver = tf.train.Saver(tf.global_variables(),
                             max_to_keep=self.cfg.MAX_TO_KEEP_CKP)

      # Build the summary operation from the last tower summaries.
      tf.summary.scalar('accuracy', accuracy)
      tf.summary.scalar('loss', loss)
      if self.cfg.WITH_REC:
        tf.summary.scalar('clf_loss', clf_loss)
        tf.summary.scalar('rec_loss', rec_loss)
      summary_op = tf.summary.merge_all()

      return global_step, train_graph, inputs, labels, input_imgs, \
          is_training, train_op, saver, summary_op, loss, accuracy, \
          clf_loss, clf_preds, rec_loss, rec_imgs


class ModelMultiTasks(ModelDistribute):

  def __init__(self,
               cfg,
               model_arch,
               restore_vars_dict=None):
    super(ModelMultiTasks, self).__init__(cfg, model_arch, restore_vars_dict)
    self.batch_size = cfg.BATCH_SIZE // cfg.GPU_NUMBER // cfg.TASK_NUMBER

  @staticmethod
  def _sum_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    This function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
                   is over individual gradients. The inner list is over the
                   gradient calculation for each tower.
        - shape: [[(grad0_gpu0, var0_gpu0), ..., (gradM_gpu0, varM_gpu0)],
                   ...,
                  [(grad0_gpuN, var0_gpuN), ..., (gradM_gpuN, varM_gpuN)]]

    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers.
    """
    sum_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Each grad_and_vars looks like:
      # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for grad, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_grad = tf.expand_dims(grad, 0)
        # Append on a 'tower' dimension which we will average over.
        grads.append(expanded_grad)

      # grads: [[grad0_gpu0], [grad0_gpu1], ..., [grad0_gpuN]]
      # Sum over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_sum(grad, 0)

      # The Variables are redundant because they are shared across towers.
      # So we will just return the first tower's pointer to the Variable.
      v = grad_and_vars[0][1]  # varI_gpu0
      grad_and_var = [grad, v]
      sum_grads.append(grad_and_var)

    # sum_grads: [[sum_grad0, var0], [sum_grad1, var1], ..., [sum_gradM, varM]]
    return sum_grads

  @staticmethod
  def _average_sum_grads(grads_sum, n_tower):
    """Calculate the average of sum_gradients.

    Args:
      grads_sum: [[sum_grad0, var0], [sum_grad1, var1], ..., [sum_gradM, varM]]

    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
    """
    avg_grads = []
    for avg_var in grads_sum:
      avg_grads.append((avg_var[0] / n_tower, avg_var[1]))

    # avg_grads: [(avg_grad0, var0), (avg_grad1, var1), ..., (avg_gradM, varM)]
    return avg_grads

  def _average_metrics_tower(self, loss_tower, acc_tower, clf_loss_tower,
                             clf_preds_tower, rec_loss_tower, rec_imgs_tower):
    """Calculate average of metrics of a tower.

    Args:
      loss_tower: final losses of each task, list
      acc_tower: accuracies of each task, list
      clf_loss_tower: classifier losses of each task, list
      clf_preds_tower: predictions of each task, list
      rec_loss_tower: reconstruction losses of each task, list
      rec_imgs_tower: reconstructed images of each task, list of 4D tensor

    Returns:
      tuple of metrics
    """
    n_task = float(len(loss_tower))

    loss_tower = tf.divide(
        tf.add_n(loss_tower), n_task, name='loss_tower')
    acc_tower = tf.divide(
        tf.add_n(acc_tower), n_task, name='acc_tower')
    clf_preds_tower = tf.concat(clf_preds_tower, axis=0, name='preds_tower')

    if self.cfg.WITH_REC:
      clf_loss_tower = tf.divide(
          tf.add_n(clf_loss_tower), n_task, name='clf_loss_tower')
      rec_loss_tower = tf.divide(
          tf.add_n(rec_loss_tower), n_task, name='rec_loss_tower')
      rec_imgs_tower = tf.concat(
          rec_imgs_tower, axis=0, name='rec_images_tower')
    else:
      clf_loss_tower, rec_loss_tower, rec_images_tower = None, None, None

    return loss_tower, acc_tower, clf_loss_tower, \
        clf_preds_tower, rec_loss_tower, rec_imgs_tower

  def _calc_on_gpu(self, gpu_idx, x_tower, y_tower, imgs_tower,
                   num_class, is_training, optimizer):

    # Split data for each tower
    x_splits_task = tf.split(
        axis=0, num_or_size_splits=self.cfg.TASK_NUMBER, value=x_tower)
    y_splits_task = tf.split(
        axis=0, num_or_size_splits=self.cfg.TASK_NUMBER, value=y_tower)
    imgs_splits_task = tf.split(
        axis=0, num_or_size_splits=self.cfg.TASK_NUMBER, value=imgs_tower)

    loss_tower, acc_tower, clf_loss_tower, clf_preds_tower, \
        rec_loss_tower, rec_imgs_tower = [], [], [], [], [], []
    # grads_tower = []
    grads_tower_sum = None
    for i in tqdm(range(self.cfg.TASK_NUMBER), ncols=100, unit=' task'):

      # Dequeues one task
      x_task, y_task, imgs_task = \
        x_splits_task[gpu_idx], y_splits_task[gpu_idx], imgs_splits_task[i]

      with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)):
        with tf.name_scope('task_%d' % i):
          # Calculate the loss for one task.
          loss_task, acc_task, clf_loss_task, clf_preds_task, \
              rec_loss_task, rec_imgs_task = self._inference(
                  x_task, y_task, imgs_task, 
                  num_class=num_class, is_training=is_training)

          # Calculate the gradients on this task.
          grads_task = optimizer.compute_gradients(loss_task)

          # Keep track of the gradients across all tasks.
          # grads_tower.append(grads_task)

          if i == 0:
            grads_tower_sum = grads_task
          else:
            grads_tower_sum = self._sum_gradients(
                [grads_tower_sum, grads_task])

          # Collect metrics of each task
          loss_tower.append(loss_task)
          acc_tower.append(acc_task)
          clf_loss_tower.append(clf_loss_task)
          clf_preds_tower.append(clf_preds_task)
          rec_loss_tower.append(rec_loss_task)
          rec_imgs_tower.append(rec_imgs_task)

    # Calculate the mean of each gradient.
    # grads_tower = self._average_gradients(grads_tower)
    grads_tower = self._average_sum_grads(
        grads_tower_sum, self.cfg.TASK_NUMBER)

    # Calculate means of metrics
    loss_tower, acc_tower, clf_loss_tower, clf_preds_tower, rec_loss_tower, \
        rec_imgs_tower = self._average_metrics_tower(
            loss_tower, acc_tower, clf_loss_tower,
            clf_preds_tower, rec_loss_tower, rec_imgs_tower)

    return grads_tower, loss_tower, acc_tower, clf_loss_tower, \
        clf_preds_tower, rec_loss_tower, rec_imgs_tower
