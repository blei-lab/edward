from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.util import get_session


class GANInference(VariationalInference):
  """Parameter estimation with GAN-style training (Goodfellow et al.,
  2014).

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.
  """
  def __init__(self, data, discriminator):
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

    Notes
    -----
    ``GANInference`` does not support model wrappers or latent
    variable inference. Note that GAN-style training also samples from
    the prior: this does not work well for latent variables that are
    shared across many data points (global variables).

    In building the computation graph for inference, the
    discriminator's parameters can be accessed with the variable scope
    "Disc".

    GANs also only work for one observed random variable in ``data``.

    Examples
    --------
    >>> z = Normal(mu=tf.zeros([100, 10]), sigma=tf.ones([100, 10]))
    >>> x = generative_network(z)
    >>>
    >>> inference = ed.GANInference({x: x_data}, discriminator)
    """
    if discriminator is None:
      raise NotImplementedError()

    self.discriminator = discriminator
    super(GANInference, self).__init__(None, data, model_wrapper=None)

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
    x_true = list(six.itervalues(self.data))[0]
    x_fake = list(six.iterkeys(self.data))[0]
    with tf.variable_scope("Disc"):
      d_true = self.discriminator(x_true)

    with tf.variable_scope("Disc", reuse=True):
      d_fake = self.discriminator(x_fake)

    loss_d = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_true), logits=d_true) + \
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(d_fake), logits=d_fake)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_fake), logits=d_fake)
    loss_d = tf.reduce_mean(loss_d)
    loss = tf.reduce_mean(loss)

    var_list_d = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
    if var_list is None:
      var_list = [v for v in tf.trainable_variables() if v not in var_list_d]

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
      if isinstance(key, tf.Tensor):
        if "Placeholder" in key.op.type:
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
