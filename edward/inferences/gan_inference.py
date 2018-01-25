from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.util import get_session


class GANInference(VariationalInference):
  """Parameter estimation with GAN-style training
  [@goodfellow2014generative].

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.

  #### Notes

  `GANInference` does not support latent variable inference. Note
  that GAN-style training also samples from the prior: this does not
  work well for latent variables that are shared across many data
  points (global variables).

  In building the computation graph for inference, the
  discriminator's parameters can be accessed with the variable scope
  "Disc".

  GANs also only work for one observed random variable in `data`.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  #### Examples

  ```python
  z = Normal(loc=tf.zeros([100, 10]), scale=tf.ones([100, 10]))
  x = generative_network(z)

  inference = ed.GANInference({x: x_data}, discriminator)
  ```
  """
  def __init__(self, data, discriminator):
    """Create an inference algorithm.

    Args:
      data: dict.
        Data dictionary which binds observed variables (of type
        `RandomVariable` or `tf.Tensor`) to their realizations (of
        type `tf.Tensor`).  It can also bind placeholders (of type
        `tf.Tensor`) used in the model to their realizations.
      discriminator: function.
        Function (with parameters) to discriminate samples. It should
        output logit probabilities (real-valued) and not probabilities
        in $[0, 1]$.
    """
    if not callable(discriminator):
      raise TypeError("discriminator must be a callable function.")

    self.discriminator = discriminator
    super(GANInference, self).__init__(None, data)

  def initialize(self, optimizer=None, optimizer_d=None,
                 global_step=None, global_step_d=None, var_list=None,
                 *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.

    Args:
      optimizer: str or tf.train.Optimizer.
        A TensorFlow optimizer, to use for optimizing the generator
        objective. Alternatively, one can pass in the name of a
        TensorFlow optimizer, and default parameters for the optimizer
        will be used.
      optimizer_d: str or tf.train.Optimizer.
        A TensorFlow optimizer, to use for optimizing the discriminator
        objective. Alternatively, one can pass in the name of a
        TensorFlow optimizer, and default parameters for the optimizer
        will be used.
      global_step: tf.Variable.
        Optional `Variable` to increment by one after the variables
        for the generator have been updated. See
        `tf.train.Optimizer.apply_gradients`.
      global_step_d: tf.Variable.
        Optional `Variable` to increment by one after the variables
        for the discriminator have been updated. See
        `tf.train.Optimizer.apply_gradients`.
      var_list: list of tf.Variable.
        List of TensorFlow variables to optimize over (in the generative
        model). Default is all trainable variables that `latent_vars`
        and `data` depend on.
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

    if self.logging:
      tf.summary.scalar("loss", self.loss,
                        collections=[self._summary_key])
      tf.summary.scalar("loss/discriminative", self.loss_d,
                        collections=[self._summary_key])
      self.summarize = tf.summary.merge_all(key=self._summary_key)

  def build_loss_and_gradients(self, var_list):
    x_true = list(six.itervalues(self.data))[0]
    x_fake = list(six.iterkeys(self.data))[0]
    with tf.variable_scope("Disc"):
      d_true = self.discriminator(x_true)

    with tf.variable_scope("Disc", reuse=True):
      d_fake = self.discriminator(x_fake)

    if self.logging:
      tf.summary.histogram("discriminator_outputs",
                           tf.concat([d_true, d_fake], axis=0),
                           collections=[self._summary_key])

    reg_terms_d = tf.losses.get_regularization_losses(scope="Disc")
    reg_terms_all = tf.losses.get_regularization_losses()
    reg_terms = [r for r in reg_terms_all if r not in reg_terms_d]

    loss_d = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_true), logits=d_true) + \
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(d_fake), logits=d_fake)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_fake), logits=d_fake)
    loss_d = tf.reduce_mean(loss_d) + tf.reduce_sum(reg_terms_d)
    loss = tf.reduce_mean(loss) + tf.reduce_sum(reg_terms)

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

    Args:
      feed_dict: dict.
        Feed dictionary for a TensorFlow session run. It is used to feed
        placeholders that are not fed during initialization.
      variables: str.
        Which set of variables to update. Either "Disc" or "Gen".
        Default is both.

    Returns:
      dict.
      Dictionary of algorithm-specific information. In this case, the
      iteration number and generative and discriminative losses.

    #### Notes

    The outputted iteration number is the total number of calls to
    `update`. Each update may include updating only a subset of
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
      raise NotImplementedError("variables must be None, 'Gen', or 'Disc'.")

    if self.debug:
      sess.run(self.op_check, feed_dict)

    if self.logging and self.n_print != 0:
      if t == 1 or t % self.n_print == 0:
        summary = sess.run(self.summarize, feed_dict)
        self.train_writer.add_summary(summary, t)

    return {'t': t, 'loss': loss, 'loss_d': loss_d}

  def print_progress(self, info_dict):
    """Print progress to output.
    """
    if self.n_print != 0:
      t = info_dict['t']
      if t == 1 or t % self.n_print == 0:
        self.progbar.update(t, {'Gen Loss': info_dict['loss'],
                                'Disc Loss': info_dict['loss_d']})


def _build_optimizer(optimizer, global_step):
  if optimizer is None and global_step is None:
    # Default optimizer always uses a global step variable.
    global_step = tf.Variable(0, trainable=False, name="global_step")

  if isinstance(global_step, tf.Variable):
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step,
                                               100, 0.9, staircase=True)
  else:
    learning_rate = 0.01

  # Build optimizer.
  if optimizer is None:
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif isinstance(optimizer, str):
    if optimizer == 'gradientdescent':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'adadelta':
      optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'momentum':
      optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    elif optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == 'ftrl':
      optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif optimizer == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
      raise ValueError('Optimizer class not found:', optimizer)
  elif not isinstance(optimizer, tf.train.Optimizer):
    raise TypeError("Optimizer must be str, tf.train.Optimizer, or None.")

  return optimizer, global_step
