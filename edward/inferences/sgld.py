from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.monte_carlo import MonteCarlo
from edward.models import RandomVariable
from edward.util import copy

try:
  from edward.models import Normal
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class SGLD(MonteCarlo):
  """Stochastic gradient Langevin dynamics (Welling and Teh, 2011).

  Notes
  -----
  In conditional inference, we infer :math:`z` in :math:`p(z, \\beta
  \mid x)` while fixing inference over :math:`\\beta` using another
  distribution :math:`q(\\beta)`.
  ``SGLD`` substitutes the model's log marginal density

  .. math::

    \log p(x, z) = \log \mathbb{E}_{q(\\beta)} [ p(x, z, \\beta) ]
                \\approx \log p(x, z, \\beta^*)

  leveraging a single Monte Carlo sample, where :math:`\\beta^* \sim
  q(\\beta)`. This is unbiased (and therefore asymptotically exact as a
  pseudo-marginal method) if :math:`q(\\beta) = p(\\beta \mid x)`.
  """
  def __init__(self, *args, **kwargs):
    """
    Examples
    --------
    >>> z = Normal(loc=0.0, scale=1.0)
    >>> x = Normal(loc=tf.ones(10) * z, scale=1.0)
    >>>
    >>> qz = Empirical(tf.Variable(tf.zeros(500)))
    >>> data = {x: np.array([0.0] * 10, dtype=np.float32)}
    >>> inference = ed.SGLD({z: qz}, data)
    """
    super(SGLD, self).__init__(*args, **kwargs)

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
    """Simulate Langevin dynamics using a discretized integrator. Its
    discretization error goes to zero as the learning rate decreases.

    Notes
    -----
    The updates assume each Empirical random variable is directly
    parameterized by ``tf.Variable``s.
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}

    # Simulate Langevin dynamics.
    learning_rate = self.step_size / tf.cast(self.t + 1, tf.float32)
    grad_log_joint = tf.gradients(self._log_joint(old_sample),
                                  list(six.itervalues(old_sample)))
    sample = {}
    for z, grad_log_p in zip(six.iterkeys(old_sample), grad_log_joint):
      qz = self.latent_vars[z]
      event_shape = qz.event_shape
      normal = Normal(loc=tf.zeros(event_shape),
                      scale=learning_rate * tf.ones(event_shape))
      sample[z] = old_sample[z] + \
          0.5 * learning_rate * tf.convert_to_tensor(grad_log_p) + \
          normal.sample()

    # Update Empirical random variables.
    assign_ops = []
    for z, qz in six.iteritems(self.latent_vars):
      variable = qz.get_variables()[0]
      assign_ops.append(tf.scatter_update(variable, self.t, sample[z]))

    # Increment n_accept.
    assign_ops.append(self.n_accept.assign_add(1))
    return tf.group(*assign_ops)

  def _log_joint(self, z_sample):
    """Utility function to calculate model's log joint density,
    log p(x, z), for inputs z (and fixed data x).

    Parameters
    ----------
    z_sample : dict
      Latent variable keys to samples.
    """
    scope = 'inference_' + str(id(self))
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    dict_swap = z_sample.copy()
    for x, qx in six.iteritems(self.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value()
        else:
          dict_swap[x] = qx

    log_joint = 0.0
    for z in six.iterkeys(self.latent_vars):
      z_copy = copy(z, dict_swap, scope=scope)
      log_joint += tf.reduce_sum(
          self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(self.data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        log_joint += tf.reduce_sum(
            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

    return log_joint
