from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.monte_carlo import MonteCarlo
from edward.models import Normal, RandomVariable
from edward.util import copy


class SGLD(MonteCarlo):
  """Stochastic gradient Langevin dynamics (Welling and Teh, 2011).
  """
  def __init__(self, latent_vars, data=None, model_wrapper=None):
    """
    Examples
    --------
    >>> z = Normal(mu=0.0, sigma=1.0)
    >>> x = Normal(mu=tf.ones(10) * z, sigma=1.0)
    >>>
    >>> qz = Empirical(tf.Variable(tf.zeros([500])))
    >>> data = {x: np.array([0.0] * 10, dtype=np.float32)}
    >>> inference = ed.SGLD({z: qz}, data)
    """
    super(SGLD, self).__init__(latent_vars, data, model_wrapper)

  def initialize(self, step_size=0.25, *args, **kwargs):
    """
    Parameters
    ----------
    step_size : float, optional
      Constant scale factor of learning rate.
    """
    self.step_size = step_size
    return super(SGLD, self).initialize(*args, **kwargs)

  def build_update(self):
    """
    Simulate Langevin dynamics using a discretized integrator. Its
    discretization error goes to zero as the learning rate decreases.
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}

    # Simulate Langevin dynamics.
    learning_rate = self.step_size / tf.cast(self.t + 1, tf.float32)
    grad_log_joint = tf.gradients(self.log_joint(old_sample),
                                  list(six.itervalues(old_sample)))
    sample = {}
    for z, qz, grad_log_p in \
        zip(six.iterkeys(self.latent_vars),
            six.itervalues(self.latent_vars),
            grad_log_joint):
      event_shape = qz.get_event_shape()
      normal = Normal(mu=tf.zeros(event_shape),
                      sigma=learning_rate * tf.ones(event_shape))
      sample[z] = old_sample[z] + 0.5 * learning_rate * grad_log_p + \
          normal.sample()

    # Update Empirical random variables.
    assign_ops = []
    variables = {x.name: x for x in
                 tf.get_default_graph().get_collection(tf.GraphKeys.VARIABLES)}
    for z, qz in six.iteritems(self.latent_vars):
      variable = variables[qz.params.op.inputs[0].op.inputs[0].name]
      assign_ops.append(tf.scatter_update(variable, self.t, sample[z]))

    # Increment n_accept.
    assign_ops.append(self.n_accept.assign_add(1))
    return tf.group(*assign_ops)

  def log_joint(self, z_sample):
    """
    Utility function to calculate model's log joint density,
    log p(x, z), for inputs z (and fixed data x).

    Parameters
    ----------
    z_sample : dict
      Latent variable keys to samples.
    """
    if self.model_wrapper is None:
      log_joint = 0.0
      for z, sample in six.iteritems(z_sample):
        z = copy(z, z_sample, scope='prior')
        log_joint += tf.reduce_sum(z.log_prob(sample))

      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          x_z = copy(x, z_sample, scope='likelihood')
          log_joint += tf.reduce_sum(x_z.log_prob(obs))
    else:
      x = self.data
      log_joint = self.model_wrapper.log_prob(x, z_sample)

    return log_joint
