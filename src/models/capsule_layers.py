from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from models.layers import ModelBase


class Capsule(ModelBase):

  def __init__(self,
               cfg,
               output_dim=None,
               output_atoms=None,
               num_routing=None,
               routing_method='v1',
               act_fn='squash',
               use_bias=False,
               share_weights=False,
               add_grads_stop=True,
               idx=0):
    """Initialize capsule layer.

    Args:
      cfg: configuration
      output_dim: number of capsules of this layer.
      output_atoms: dimensions of vectors of capsules.
      num_routing: number of dynamic routing iteration.
      routing_method: version of dynamic routing implement.
      act_fn: string, activation function.
      use_bias: bool, if add biases.
      share_weights: if share weights matrix.
      add_grads_stop: add gradients stops.
      idx: int, index of layer.
    """
    super(Capsule, self).__init__()
    self.cfg = cfg
    self.output_dim = output_dim
    self.output_atoms = output_atoms
    self.num_routing = num_routing
    self.routing_method = routing_method
    self.act_fn = act_fn
    self.use_bias = use_bias
    self.share_weights = share_weights
    self.add_grads_stop = add_grads_stop
    self.idx = idx

  def _dynamic_routing(self, inputs, weights, biases, act_fn):
    """Dynamic routing.

    Args:
      inputs: input tensor
        - shape: [batch_size, input_dim, input_atoms]
      weights: weights matrix
      biases: b_ij
      act_fn: activation function

    Returns:
      output tensor
        - shape: [batch_size, output_dim, output_atoms]
    """
    batch_size, input_dim, input_atoms = inputs.get_shape().as_list()

    # inputs shape: (batch_size, input_dim, output_dim, input_atoms, 1)
    inputs = tf.reshape(
        inputs, shape=[batch_size, input_dim, 1, input_atoms, 1])
    inputs = tf.tile(inputs, [1, 1, self.output_dim, 1, 1])

    # weights: (batch_size, input_dim, output_dim, output_atoms, input_atoms)
    weights = tf.tile(weights, [batch_size, 1, 1, 1, 1])

    # ( , , , output_atoms, input_atoms) x ( , , , vec_dim_i, 1)
    # u_hat: (batch_size, input_dim, output_dim, output_atoms, 1)
    u_hat = tf.matmul(weights, inputs)

    # u_hat_stop
    # Do not transfer the gradient of u_hat_stop during back-propagation
    if self.add_grads_stop:
      u_hat_stop = tf.stop_gradient(u_hat)
    else:
      u_hat_stop = u_hat

    b_ij = tf.fill(
        tf.stack([batch_size, input_dim, self.output_dim, 1, 1]), 0.0)

    use_bias = self.use_bias
    def _sum_and_activate(_u_hat, _c_ij):
      """Get sum of vectors and apply activation function."""
      # Using u_hat but not u_hat_stop in order to transfer gradients.
      # _s_j: (batch_size, output_dim, output_atoms, 1)
      _s_j = tf.reduce_sum(tf.multiply(_u_hat, _c_ij), axis=1)

      if use_bias:
        _s_j = _s_j + tf.expand_dims(biases, -1)

      # shape: (batch_size, output_dim, output_atoms, 1)
      return act_fn(_s_j)

    votes = None
    for iter_route in range(self.num_routing):

      # Calculate c_ij for every epoch
      # c_ij: (batch_size, input_dim, output_dim, 1, 1)
      c_ij = tf.nn.softmax(b_ij, dim=2)

      # Applying back-propagation at last epoch.
      if iter_route == self.num_routing - 1:
        # Do not transfer the gradient of c_ij_stop during back-propagation.
        if self.add_grads_stop:
          c_ij_stop = tf.stop_gradient(c_ij)
        else:
          c_ij_stop = c_ij

        # Using u_hat but not u_hat_stop in order to transfer gradients.
        votes = _sum_and_activate(u_hat, c_ij_stop)

      # Do not apply back-propagation if it is not last epoch.
      else:
        # Using u_hat_stop so that the gradient will not be transferred to
        # routing processes.
        votes = _sum_and_activate(u_hat_stop, c_ij)

        # votes_reshaped: (batch_size, input_dim, output_dim, 1, output_atoms)
        votes_reshaped = tf.reshape(
            votes, shape=[-1, 1, self.output_dim, 1, self.output_atoms])
        votes_reshaped = tf.tile(votes_reshaped, [1, input_dim, 1, 1, 1])

        # ( , , , 1, output_atoms) x ( , , , output_atoms, 1)
        # delta_b_ij shape: (batch_size, input_dim, output_dim, 1)
        delta_b_ij = tf.matmul(votes_reshaped, u_hat_stop)

        # Updating: b_ij <- b_ij + v_j x u_ij
        # b_ij shape: (batch_size, input_dim, output_dim, 1, 1)
        b_ij = tf.add(b_ij, delta_b_ij)

    # votes: (batch_size, output_dim, output_atoms)
    votes = tf.squeeze(votes, axis=-1)
    return votes

  def _dynamic_routing_v2(self, inputs, weights, biases, act_fn):
    """Dynamic routing.

    Args:
      inputs: input tensor
        - shape: [batch_size, input_dim, input_atoms]
      weights: weights matrix
      biases: b_ij
      act_fn: activation function

    Returns:
      output tensor
        - shape: [batch_size, output_dim, output_atoms]
    """
    num_routing = self.num_routing
    output_dim = self.output_dim
    output_atoms = self.output_atoms
    use_bias = self.use_bias
    batch_size, input_dim, input_atoms = inputs.get_shape().as_list()

    inputs = tf.reshape(
        inputs, shape=[batch_size, input_dim, 1, input_atoms, 1])
    inputs = tf.tile(inputs, [1, 1, self.output_dim, 1, 1])

    weights = tf.tile(weights, [batch_size, 1, 1, 1, 1])
    u_hat = tf.matmul(weights, inputs)

    b_ij = tf.fill(
        tf.stack([batch_size, input_dim, self.output_dim, 1, 1]), 0.0)

    def _sum_and_activate(_c_ij):
      """Get sum of vectors and apply activation function."""
      _s_j = tf.reduce_sum(tf.multiply(u_hat, _c_ij), axis=1)
      if use_bias:
        _s_j = _s_j + tf.expand_dims(biases, -1)
      return act_fn(_s_j)

    def _body(i_route_, b_ij_, activations_):
      """Routing while loop."""
      c_ij_ = tf.nn.softmax(b_ij_, dim=2)
      votes_ = _sum_and_activate(c_ij_)
      activations_ = activations_.write(i_route_, votes_)
      votes_reshaped = tf.reshape(
          votes_, shape=[-1, 1, output_dim, 1, output_atoms])
      votes_reshaped = tf.tile(votes_reshaped, [1, input_dim, 1, 1, 1])
      b_ij_ = tf.add(b_ij_, tf.matmul(votes_reshaped, u_hat))
      return i_route_ + 1, b_ij_, activations_

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    i_route = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i_route_, b_ij_, activations_: i_route_ < num_routing,
        _body,
        loop_vars=[i_route, b_ij, activations],
        swap_memory=True)

    votes = tf.squeeze(activations.read(num_routing - 1), axis=-1)
    return votes

  def __call__(self, inputs):
    """Apply dynamic routing.

    Args:
      inputs: input tensor
        - shape: [batch, input_dim, input_atoms]

    Returns:
      output tensor
        - shape: [batch, output_dim, output_atoms]
    """
    with tf.variable_scope('caps_{}'.format(self.idx)):

      batch_size, input_dim, input_atoms = inputs.get_shape().as_list()

      # weights_initializer = tf.contrib.layers.xavier_initializer()
      weights_initializer = tf.truncated_normal_initializer(
          stddev=0.01, dtype=tf.float32)
      biases_initializer = tf.zeros_initializer()

      if self.share_weights:
        weights_shape = [1, 1, self.output_dim, self.output_atoms, input_atoms]
      else:
        weights_shape = \
            [1, input_dim, self.output_dim, self.output_atoms, input_atoms]
      weights, biases = self._get_variables(
          use_bias=self.use_bias,
          weights_shape=weights_shape,
          biases_shape=[self.output_dim, self.output_atoms],
          weights_initializer=weights_initializer,
          biases_initializer=biases_initializer,
          store_on_cpu=self.cfg.VAR_ON_CPU
      )
      if self.share_weights:
        weights = tf.tile(weights, [1, input_dim, 1, 1, 1])

      if self.routing_method == 'v1':
        dr_algorithm = self._dynamic_routing
      elif self.routing_method == 'v2':
        dr_algorithm = self._dynamic_routing_v2
      else:
        raise ValueError('Wrong dynamic routing version!')

      self.output = dr_algorithm(
          inputs=inputs,
          weights=weights,
          biases=biases,
          act_fn=self._get_act_fn(self.act_fn)
      )

    return self.output


class ConvSlimCapsule(ModelBase):

  def __init__(self,
               cfg,
               output_dim=None,
               output_atoms=None,
               kernel_size=None,
               stride=None,
               padding='SAME',
               conv_act_fn='relu',
               caps_act_fn='squash',
               use_bias=True,
               idx=0):
    """Generate a Capsule layer using convolution kernel.

    Args:
      cfg: configuration
      output_dim: number of capsules of this layer.
      output_atoms: dimensions of vectors of capsules.
      kernel_size: size of convolution kernel.
      stride: stride of convolution kernel.
      padding: padding type of convolution kernel.
      conv_act_fn: activation function of convolution.
      caps_act_fn: activation function of capsule.
      use_bias: bool, if add biases.
      idx: int, index of layer.
    """
    super(ConvSlimCapsule, self).__init__()
    self.cfg = cfg
    self.output_dim = output_dim
    self.output_atoms = output_atoms
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.conv_act_fn = conv_act_fn
    self.caps_act_fn = caps_act_fn
    self.use_bias = use_bias
    self.idx = idx

  def __call__(self, inputs):
    """Convert a convolution layer to capsule layer.

    Args:
      inputs: input tensor
        - NHWC: [batch, input_height, input_width, input_channels]
        - NCHW: [batch, input_channels, input_height, input_width]

    Returns:
      tensor of capsules
        - [batch, output_dim * output_height * output_width, output_atoms]
    """
    with tf.variable_scope('caps_{}'.format(self.idx)):

      if self.cfg.DATA_FORMAT == 'NHWC':
        batch_size, input_height, input_width, input_channels = \
          inputs.get_shape().as_list()
      elif self.cfg.DATA_FORMAT == 'NCHW':
        batch_size, input_channels, input_height, input_width = \
          inputs.get_shape().as_list()
      else:
        raise ValueError('Wrong data format!')

      # Convolution layer
      weights_initializer = tf.contrib.layers.xavier_initializer()
      biases_initializer = tf.zeros_initializer()
      # weights_initializer = tf.truncated_normal_initializer(
      #     stddev=0.1, dtype=tf.float32)
      # biases_initializer = tf.constant_initializer(0.1) \
      #     if self.use_bias else None

      weights, biases = self._get_variables(
          use_bias=self.use_bias,
          weights_shape=[self.kernel_size, self.kernel_size,
                         input_channels, self.output_dim * self.output_atoms],
          biases_shape=[self.output_dim * self.output_atoms],
          weights_initializer=weights_initializer,
          biases_initializer=biases_initializer,
          store_on_cpu=self.cfg.VAR_ON_CPU
      )

      caps = tf.nn.conv2d(
          input=inputs,
          filter=weights,
          strides=self._get_strides_list(self.stride, self.cfg.DATA_FORMAT),
          padding=self.padding,
          data_format=self.cfg.DATA_FORMAT
      )

      if self.use_bias:
        caps = tf.nn.bias_add(caps, biases, data_format=self.cfg.DATA_FORMAT)

      if self.conv_act_fn is not None:
        caps = self._get_act_fn(self.conv_act_fn)(caps)

      if self.cfg.DATA_FORMAT == 'NHWC':
        # caps shape:
        # (batch_size, output_height, output_width,
        #  self.output_dim * self.output_atoms)
        pass
      elif self.cfg.DATA_FORMAT == 'NCHW':
        # caps shape:
        # (batch_size, self.output_dim * self.output_atoms,
        #  output_height, output_width)
        caps = tf.transpose(caps, [0, 2, 3, 1])
      else:
        raise ValueError('Wrong data format!')

      # Reshape and generating a capsule layer
      # reshaped caps shape:
      # (batch_size, output_dim * output_height * output_width, output_atoms)
      caps = tf.reshape(caps, [batch_size, -1, self.output_atoms])

      # Applying activation function
      act_fn_caps = self._get_act_fn(self.caps_act_fn)
      self.output = act_fn_caps(caps)
      # caps_activated shape:
      # (batch_size, output_dim * output_height * output_width,
      #  output_atoms, output_atoms)

      return self.output


class CapsuleV2(ModelBase):

  def __init__(self,
               cfg,
               output_dim,
               output_atoms,
               num_routing=3,
               leaky=False,
               act_fn='squash_v2',
               use_bias=False,
               idx=0):
    """Single convolution layer

    Args:
      cfg: configuration
      output_dim: scalar, number of capsules in this layer.
      output_atoms: scalar, number of units in each capsule of output layer.
      num_routing: scalar, Number of routing iterations.
      leaky: boolean, if set use leaky routing.
      act_fn: string, activation function.
      use_bias: bool, if add biases.
      idx: int, index of layer.
    """
    super(CapsuleV2, self).__init__()
    self.cfg = cfg
    self.output_dim = output_dim
    self.output_atoms = output_atoms
    self.num_routing = num_routing
    self.leaky = leaky
    self.act_fn = act_fn
    self.use_bias = use_bias
    self.idx = idx

  @staticmethod
  def _leaky_routing(logits, output_dim):
    """Adds extra dimension to routing logits.

    This enables active capsules to be routed to the extra dim if they are not a
    good fit for any of the capsules in layer above.

    Args:
      logits: The original logits. shape is
        [input_capsule_num, output_capsule_num] if fully connected. Otherwise,
        it has two more dimensions.
      output_dim: The number of units in the second dimension of logits.

    Returns:
      Routing probabilities for each pair of capsules. Same shape as logits.
    """

    # leak is a zero matrix with same shape as logits except dim(2) = 1 because
    # of the reduce_sum.
    leak = tf.zeros_like(logits, optimize=True)
    leak = tf.reduce_sum(leak, axis=2, keep_dims=True)
    leaky_logits = tf.concat([leak, logits], axis=2)
    leaky_routing = tf.nn.softmax(leaky_logits, dim=2)
    return tf.split(leaky_routing, [1, output_dim], 2)[1]

  def _dynamic_routing(self,
                       votes,
                       biases,
                       logit_shape,
                       num_dims,
                       input_dim,
                       output_dim,
                       act_fn,
                       num_routing,
                       leaky=False):
    """Sums over scaled votes and applies squash to compute the activations.

    Iteratively updates routing logits (scales) based on the similarity between
    the activation of this layer and the votes of the layer below.

    Args:
      votes: tensor, The transformed outputs of the layer below.
      biases: tensor, Bias tf_variable.
      logit_shape: tensor, shape of the logit to be initialized.
      num_dims: scalar, number of dimensions in votes. For fully connected
        capsule it is 4, for convolution 6.
      input_dim: scalar, number of capsules in the input layer.
      output_dim: scalar, number of capsules in the output layer.
      act_fn: activation function.
      num_routing: scalar, Number of routing iterations.
      leaky: boolean, if set use leaky routing.

    Returns:
      The activation tensor of the output layer after num_routing iterations.
    """
    # votes shape: [batch, input_dim, output_dim, output_atoms]

    votes_t_shape = [3, 0, 1, 2]
    for i in range(num_dims - 4):
      votes_t_shape += [i + 4]  # CONV: votes_t_shape - [3, 0, 1, 2, 4, 5]
    r_t_shape = [1, 2, 3, 0]
    for i in range(num_dims - 4):
      r_t_shape += [i + 4]  # CONV: r_t_shape - [1, 2, 3, 0, 4, 5]

    # votes_trans: [output_atoms, batch, input_dim, output_dim]
    votes_trans = tf.transpose(votes, votes_t_shape)

    use_bias = self.use_bias

    def _body(i_route, logits_, activations_):
      """Routing while loop."""
      # logits_: [batch, input_dim, output_dim]
      if leaky:
        route = self._leaky_routing(logits_, output_dim)
      else:
        route = tf.nn.softmax(logits_, dim=2)

      # route: [batch, input_dim, output_dim]
      # pre_act_unrolled: [output_atoms, batch, input_dim, output_dim]
      pre_act_unrolled = route * votes_trans

      # pre_act_trans: [batch, input_dim, output_dim, output_atoms]
      pre_act_trans = tf.transpose(pre_act_unrolled, r_t_shape)

      # pre_act: [batch, output_dim, output_atoms]
      if use_bias:
        pre_act = tf.reduce_sum(pre_act_trans, axis=1) + biases
      else:
        pre_act = tf.reduce_sum(pre_act_trans, axis=1)

      # activated: [batch, output_dim, output_atoms]
      activated = act_fn(pre_act)
      activations_ = activations_.write(i_route, activated)

      # act_3d: [batch, 1, output_dim, output_atoms]
      act_3d = tf.expand_dims(activated, 1)

      # act_tiled: [batch, input_dim, output_dim, output_atoms]
      tile_shape = list(np.ones(num_dims, dtype=np.int32))
      tile_shape[1] = input_dim
      act_tiled = tf.tile(act_3d, tile_shape)

      # distances: [batch, input_dim, output_dim]
      distances = tf.reduce_sum(votes * act_tiled, axis=3)
      logits_ += distances

      return i_route + 1, logits_, activations_

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)

    logits = tf.fill(logit_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i_route, logits_, activations_: i_route < num_routing,
        _body,
        loop_vars=[i, logits, activations],
        swap_memory=True)

    return activations.read(num_routing - 1)

  def __call__(self, inputs):
    """Single full-connected layer

    Args:
      inputs: tensor, activation output of the layer below of shape
      `[batch, input_dim, input_atoms]`.

    Returns:
      Tensor of activations for this layer of shape
      `[batch, output_dim, output_atoms]`.
    """
    with tf.variable_scope('caps_{}'.format(self.idx)):

        batch_size, input_dim, input_atoms = inputs.get_shape().as_list()

        # weights_initializer = tf.truncated_normal_initializer(
        #     stddev=0.01, dtype=tf.float32)
        # biases_initializer = tf.zeros_initializer()
        weights_initializer = tf.truncated_normal_initializer(
            stddev=0.1, dtype=tf.float32)
        biases_initializer = tf.constant_initializer(0.1)

        weights, biases = self._get_variables(
            use_bias=self.use_bias,
            weights_shape=[input_dim, input_atoms,
                           self.output_dim * self.output_atoms],
            biases_shape=[self.output_dim, self.output_atoms],
            weights_initializer=weights_initializer,
            biases_initializer=biases_initializer,
            store_on_cpu=self.cfg.VAR_ON_CPU
        )

        with tf.name_scope('Wx_plus_b'):
            # Depth-wise matmul: [b, d, c] ** [d, c, o_c] = [b, d, o_c]
            # To do this: tile input, do element-wise multiplication and reduce
            # sum over input_atoms dimension.

            # [batch, input_dim, input_atoms, output_dim *output_atoms]
            input_tiled = tf.tile(
                tf.expand_dims(inputs, -1),
                [1, 1, 1, self.output_dim * self.output_atoms])

            # [batch, input_dim, output_dim * output_atoms]
            votes = tf.reduce_sum(input_tiled * weights, axis=2)

            # [batch, input_dim, output_dim, output_atoms]
            votes_reshaped = tf.reshape(
                votes, [-1, input_dim, self.output_dim, self.output_atoms])

        with tf.name_scope('routing'):
            logit_shape = tf.stack([batch_size, input_dim, self.output_dim])
            self.output = self._dynamic_routing(
                votes=votes_reshaped,
                biases=biases,
                logit_shape=logit_shape,
                num_dims=4,
                input_dim=input_dim,
                output_dim=self.output_dim,
                act_fn=self._get_act_fn(self.act_fn),
                num_routing=self.num_routing,
                leaky=self.leaky)

        return self.output


class ConvSlimCapsuleV2(CapsuleV2):

  def __init__(self,
               cfg,
               output_dim,
               output_atoms,
               num_routing=3,
               leaky=False,
               kernel_size=5,
               stride=2,
               padding='SAME',
               conv_act_fn=None,
               caps_act_fn='squash_v2',
               use_bias=True,
               idx=0):
    """Builds a slim convolutional capsule layer.

      This layer performs 2D convolution given 5D input tensor of shape
      `[batch, input_dim, input_atoms, input_height, input_width]`. Then refines
      the votes with routing and applies Squash non linearity for each capsule.

      Each capsule in this layer is a convolutional unit and shares its kernel
      over the position grid and different capsules of layer below. Therefore,
      number of trainable variables in this layer is:

      kernel: [kernel_size, kernel_size, input_atoms, output_dim * output_atoms]
      bias: [output_dim, output_atoms]

      Output of a conv2d layer is a single capsule with channel number of atoms.
      Therefore conv_slim_capsule is suitable to be added on top of a conv2d
      layer with num_routing=1, input_dim=1 and input_atoms=conv_channels.

    Args:
      cfg: configuration
      output_dim: scalar, number of capsules in this layer.
      output_atoms: scalar, number of units in each capsule of output layer.
      num_routing: scalar, Number of routing iterations.
      leaky: boolean, if set use leaky routing.
      kernel_size: scalar, convolutional kernels are [kernel_size, kernel_size].
      stride: scalar, stride of the convolutional kernel.
      padding: 'SAME' or 'VALID', padding mechanism for convolutional kernels.
      conv_act_fn: activation function of convolution.
      caps_act_fn: activation function of capsule.
      use_bias: bool, if add biases.
      idx: int, index of layer.
    """
    super(ConvSlimCapsuleV2, self).__init__(
        cfg=cfg,
        output_dim=output_dim,
        output_atoms=output_atoms,
        num_routing=num_routing,
        leaky=leaky,
        act_fn=caps_act_fn,
        use_bias=use_bias,
        idx=idx
    )
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.conv_act_fn = conv_act_fn
    self.caps_act_fn = caps_act_fn

  def _depthwise_conv3d(self, input_tensor, kernel):
    """Performs 2D convolution given a 5D input tensor.

    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width]` squeezes the
    first two dimensions to get a 4D tensor as the input of tf.nn.conv2d. Then
    splits the first dimension and the last dimension and returns the 6D
    convolution output.

    Args:
      input_tensor: tensor, of rank 5. Last two dimensions representing height
        and width position grid.
        - shape: [batch, 1, 256, height, width]
      kernel: Tensor, convolutional kernel variables.

    Returns:
      6D Tensor output of a 2D convolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width]`,
        the convolution output shape and the input shape.
        If padding is 'SAME', out_height = in_height and out_width = in_width.
        Otherwise, height and width is adjusted with same rules as 'VALID' in
        tf.nn.conv2d.
    """
    with tf.name_scope('conv'):
      batch_size, input_dim, input_atoms, \
          input_height, input_width = input_tensor.get_shape().as_list()

      # Reshape input_tensor to 4D by merging first two dimensions.
      # tf.nn.conv2d only accepts 4D tensors.
      input_tensor_reshaped = tf.reshape(input_tensor, [
        batch_size * input_dim, input_atoms, input_height, input_width
      ])
      input_tensor_reshaped.set_shape(
          (None, input_atoms, input_height, input_width))
      conv = tf.nn.conv2d(
          input_tensor_reshaped,
          kernel,
          [1, 1, self.stride, self.stride],
          padding=self.padding,
          data_format='NCHW')
      conv_shape = tf.shape(conv)

      if self.conv_act_fn is not None:
        act_fn_conv = self._get_act_fn(self.conv_act_fn)
        conv = act_fn_conv(conv)

      _, _, conv_height, conv_width = conv.get_shape().as_list()
      # Reshape back to 6D by splitting first dimension to batch and input_dim
      # and splitting second dimension to output_dim and output_atoms.

      conv_reshaped = tf.reshape(conv, [
        batch_size, input_dim, self.output_dim,
        self.output_atoms, conv_shape[2], conv_shape[3]
      ])
      conv_reshaped.set_shape((
        None, input_dim, self.output_dim,
        self.output_atoms, conv_height, conv_width
      ))
      return conv_reshaped, conv_shape

  def __call__(self, inputs):
    """Single full-connected layer

    Args:
      inputs: tensor, 5D input tensor of shape
      `[batch, input_dim, input_atoms, input_height, input_width]`. Then refines
      the votes with routing and applies Squash non linearity for each capsule.

    Returns:
      Tensor of activations for this layer of shape
      `[batch, output_dim, output_atoms, out_height, out_width]`. If padding is
      'SAME', out_height = in_height and out_width = in_width. Otherwise, height
      and width is adjusted with same rules as 'VALID' in tf.nn.conv2d.
    """
    with tf.variable_scope('caps_{}'.format(self.idx)):

      batch_size, input_dim, input_atoms, _, _ = inputs.get_shape().as_list()

      # weights_initializer = tf.contrib.layers.xavier_initializer()
      # biases_initializer = tf.zeros_initializer()
      weights_initializer = tf.truncated_normal_initializer(
          stddev=0.1, dtype=tf.float32)
      biases_initializer = tf.constant_initializer(0.1)

      weights, biases = self._get_variables(
          use_bias=self.use_bias,
          weights_shape=[self.kernel_size, self.kernel_size,
                         input_atoms, self.output_dim * self.output_atoms],
          biases_shape=[self.output_dim, self.output_atoms, 1, 1],
          weights_initializer=weights_initializer,
          biases_initializer=biases_initializer,
          store_on_cpu=self.cfg.VAR_ON_CPU
      )

      votes, votes_shape = self._depthwise_conv3d(inputs, weights)

      with tf.name_scope('routing'):
        logit_shape = tf.stack([
          batch_size, input_dim, self.output_dim, votes_shape[2], votes_shape[3]
        ])
        if self.use_bias:
          biases = tf.tile(biases, [1, 1, votes_shape[2], votes_shape[3]])
        self.output = self._dynamic_routing(
            votes=votes,
            biases=biases,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=input_dim,
            output_dim=self.output_dim,
            act_fn=self._get_act_fn(self.caps_act_fn),
            num_routing=self.num_routing,
            leaky=self.leaky)

      return self.output


class Mask(ModelBase):

  def __init__(self, labels):
    """Get masked tensor.

    Args:
      labels: labels of inputs tensor
    """
    super(Mask, self).__init__()
    self.labels = labels

  def __call__(self, inputs):
    """Reshape a tensor.

    Args:
      inputs: input tensor
        - shape: [batch, input_dim, input_atoms]

    Returns:
      masked tensor
    """
    self.output = tf.reduce_sum(
        tf.multiply(inputs, tf.expand_dims(self.labels, axis=-1)), axis=1)
    return self.output


class Capsule5Dto3D(ModelBase):

  def __init__(self):
    """Convert a capsule output tensor to 3D tensor"""
    super(Capsule5Dto3D, self).__init__()

  def __call__(self, inputs):
    """Convert a capsule output tensor to 3D tensor.

    Args:
      inputs: 5D tensor
        - shape: [batch, input_dim, input_atoms, input_height, input_width]

    Returns:
      output tensor
        - shape: [batch, new_input_dim, input_atoms]
    """
    batch_size, _, atoms, _, _ = inputs.get_shape().as_list()
    output = tf.transpose(inputs, [0, 1, 3, 4, 2])
    self.output = tf.reshape(output, [batch_size, -1, atoms])
    return self.output


class Capsule4Dto5D(ModelBase):

  def __init__(self, data_format):
    """Convert a conv2d output tensor to 5D tensor.

    Args:
      data_format: data format of images, 'NCHW' or 'NHWC'
    """
    super(Capsule4Dto5D, self).__init__()
    self.data_format = data_format

  def __call__(self, inputs):
    """Convert a conv2d output tensor to 5D tensor.

    Args:
      inputs: 4D tensor
        - NHWC: [batch, input_height, input_width, input_channels]
        - NCHW: [batch, input_channels, input_height, input_width]

    Returns:
      output: 5D tensor of shape
        - [batch, input_dim, input_atoms, input_height, input_width]
        - [batch, 1, input_channels, input_height, input_width]
    """
    if self.data_format == 'NHWC':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    elif self.data_format == 'NCHW':
      pass
    else:
      raise ValueError('Wrong data format!')

    self.output = tf.expand_dims(inputs, 1)
    return self.output
