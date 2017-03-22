from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.util import get_session


class ALI(VariationalInference):
  """Adversarially Learned Inference (Dumoulin et al., 2016) or
  Bidirectional Generative Adversarial Networks (Donahue et al., 2016)
  for joint learning of generator and inference networks.

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.
  """
  def __init__(self, data, discriminator, encoder, decoder):
    """
    Parameters
    ----------
    data : dict
      Data dictionary which binds observed variables (of type
      ``RandomVariable`` or ``tf.Tensor``) to their realizations (of
      type ``tf.Tensor``).  It can also bind placeholders (of type
      ``tf.Tensor``) used in the model to their realizations.
    discriminator : function
      Function (with parameters) to discriminate samples. It should
      output logit probabilities (real-valued) and not probabilities
      in [0, 1].
    encoder : function
      Function (with parameters) to generate latent variables from
      observed data, approximates q(z|x).
    decoder : function
      Function (with parameters) to generate observed data from
      latent variables, approximates p(x|z).

    Notes
    -----
    ``ALI`` matches a mapping from data to latent variables and a
    mapping from latent variables to data through a joint
    discriminator. The encoder approximates the posterior p(z|x)
    when the network is stochastic.

    In building the computation graph for inference, the
    discriminator's parameters can be accessed with the variable scope
    "Disc".
    In building the computation graph for inference, the
    encoder and decoder parameters can be accessed with the variable scope
    "Gen".

    Examples
    --------
    >>> zs = ed.models.Normal(mu=tf.zeros([M, d]), sigma=tf.ones([M, d]))
    >>> xf = gen_data(zs)
    >>> xs = ed.models.Empirical(data)
    >>> zf = gen_latent(xs._sample_n(M))
    >>> inference = ed.ALI({xf: x_data, zf: z_samples},
                            discriminator, encoder, decoder)
    """
    if discriminator is None:
      raise NotImplementedError()

    self.discriminator = discriminator
    self.encoder = encoder
    self.decoder = decoder
    super(ALI, self).__init__(None, data)

  def initialize(self, optimizer=None, optimizer_d=None,
                 global_step=None, global_step_d=None, var_list=None,
                 *args, **kwargs):
    """Initialize variational inference.

    Parameters
    ----------
    optimizer : str or tf.train.Optimizer, optional
      A TensorFlow optimizer, to use for optimizing the generator
      objective. Alternatively, one can pass in the name of a
      TensorFlow optimizer, and default parameters for the optimizer
      will be used.
    optimizer_d : str or tf.train.Optimizer, optional
      A TensorFlow optimizer, to use for optimizing the discriminator
      objective. Alternatively, one can pass in the name of a
      TensorFlow optimizer, and default parameters for the optimizer
      will be used.
    global_step : tf.Variable, optional
      Optional ``Variable`` to increment by one after the variables
      for the generator have been updated. See
      ``tf.train.Optimizer.apply_gradients``.
    global_step_d : tf.Variable, optional
      Optional ``Variable`` to increment by one after the variables
      for the discriminator have been updated. See
      ``tf.train.Optimizer.apply_gradients``.
    var_list : list of tf.Variable, optional
      List of TensorFlow variables to optimize over (in the generative
      model). Default is all trainable variables that ``latent_vars``
      and ``data`` depend on.
    """
    # call grandparent's method; avoid parent (VariationalInference)
    super(VariationalInference, self).initialize(*args, **kwargs)

    self.loss, grads_and_vars, self.loss_d, grads_and_vars_d = \
        self.build_loss_and_gradients(var_list)

    optimizer, global_step = _build_optimizer(optimizer, global_step)
    optimizer_d, global_step_d = _build_optimizer(optimizer_d, global_step_d)

    self.train = optimizer.apply_gradients(grads_and_vars,
                                           global_step=global_step)
    self.train_d = optimizer_d.apply_gradients(grads_and_vars_d,
                                               global_step=global_step_d)

  def build_loss_and_gradients(self, var_list):
    # Does not use feed_dict's keys, since the fakes are generated
    # by the feed_dict values. Wanted to keep implementation close
    # to original gan_inference, hence this may not be the best
    # implenetation.
    x_true = list(six.itervalues(self.data))[0]
    z_true = list(six.itervalues(self.data))[1]
    with tf.variable_scope("Gen"):
        x_fake = self.decoder(z_true)
        z_fake = self.encoder(x_true)
    with tf.variable_scope("Disc"):
        # xtzf := x_true, z_fake
        d_xtzf = self.discriminator(x_true, z_fake)
    with tf.variable_scope("Disc", reuse=True):
        # xfzt := x_fake, z_true
        d_xfzt = self.discriminator(x_fake, z_true)

    loss_d = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(d_xfzt), logits=d_xfzt) + \
      tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(d_xtzf), logits=d_xtzf)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(d_xfzt), logits=d_xfzt) + \
      tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_xtzf), logits=d_xtzf)
    loss_d = tf.reduce_mean(loss_d)
    loss = tf.reduce_mean(loss)

    var_list_d = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
    var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="Gen")

    grads_d = tf.gradients(loss_d, var_list_d)
    grads = tf.gradients(loss, var_list)
    grads_and_vars_d = list(zip(grads_d, var_list_d))
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars, loss_d, grads_and_vars_d

  def update(self, feed_dict=None, variables=None):
    """Run one iteration of optimization.

    Parameters
    ----------
    feed_dict : dict, optional
      Feed dictionary for a TensorFlow session run. It is used to feed
      placeholders that are not fed during initialization.
    variables : str, optional
      Which set of variables to update. Either "Disc" or "Gen".
      Default is both.

    Returns
    -------
    dict
      Dictionary of algorithm-specific information. In this case, the
      iteration number and generative and discriminative losses.

    Notes
    -----
    The outputted iteration number is the total number of calls to
    ``update``. Each update may include updating only a subset of
    parameters.
    """
    if feed_dict is None:
      feed_dict = {}

    for key, value in six.iteritems(self.data):
      if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
        feed_dict[key] = value

    sess = get_session()
    if variables is None:
      _, _, t, loss, loss_d = sess.run(
          [self.train, self.train_d, self.increment_t, self.loss, self.loss_d],
          feed_dict)
    elif variables == "Gen":
      _, t, loss = sess.run(
          [self.train, self.increment_t, self.loss], feed_dict)
      loss_d = 0.0
    elif variables == "Disc":
      _, t, loss_d = sess.run(
          [self.train_d, self.increment_t, self.loss_d], feed_dict)
      loss = 0.0
    else:
      raise NotImplementedError()

    if self.debug:
      sess.run(self.op_check)

    if self.logging and self.n_print != 0:
      if t == 1 or t % self.n_print == 0:
        if self.summarize is not None:
          summary = sess.run(self.summarize, feed_dict)
          self.train_writer.add_summary(summary, t)

    return {'t': t, 'loss': loss, 'loss_d': loss_d}

  def print_progress(self, info_dict):
    """Print progress to output.
    """
    if self.n_print != 0:
      t = info_dict['t']
      if t == 1 or t % self.n_print == 0:
        loss = info_dict['loss']
        loss_d = info_dict['loss_d']
        string = 'Iteration {0}'.format(str(t).rjust(len(str(self.n_iter))))
        string += ' [{0}%]'.format(str(int(t / self.n_iter * 100)).rjust(3))
        string += ': Gen Loss = {0:.3f}'.format(loss)
        string += ': Disc Loss = {0:.3f}'.format(loss_d)
        print(string)


def _build_optimizer(optimizer, global_step):
  if optimizer is None:
    # Use ADAM with a decaying scale factor.
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step,
                                               100, 0.9, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif isinstance(optimizer, str):
    if optimizer == 'gradientdescent':
      optimizer = tf.train.GradientDescentOptimizer(0.01)
    elif optimizer == 'adadelta':
      optimizer = tf.train.AdadeltaOptimizer()
    elif optimizer == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(0.01)
    elif optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
    elif optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer()
    elif optimizer == 'ftrl':
      optimizer = tf.train.FtrlOptimizer(0.01)
    elif optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(0.01)
    else:
      raise ValueError('Optimizer class not found:', optimizer)
  elif not isinstance(optimizer, tf.train.Optimizer):
    raise TypeError()

  return optimizer, global_step
