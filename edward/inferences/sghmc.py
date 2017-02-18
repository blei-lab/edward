from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.monte_carlo import MonteCarlo
from edward.models import Normal, RandomVariable, Empirical
from edward.util import copy


class SGHMC(MonteCarlo):
  """Stochastic gradient Hamiltonian Monte Carlo (Chen et al., 2014).

  Notes
  -----
  In conditional inference, we infer :math:`z` in :math:`p(z, \\beta
  \mid x)` while fixing inference over :math:`\\beta` using another
  distribution :math:`q(\\beta)`.
  ``SGHMC`` substitutes the model's log marginal density

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
    >>> z = Normal(mu=0.0, sigma=1.0)
    >>> x = Normal(mu=tf.ones(10) * z, sigma=1.0)
    >>>
    >>> qz = Empirical(tf.Variable(tf.zeros([500])))
    >>> data = {x: np.array([0.0] * 10, dtype=np.float32)}
    >>> inference = ed.SGHMC({z: qz}, data)
    """
    super(SGHMC, self).__init__(*args, **kwargs)

  def initialize(self, step_size=0.25, friction=0.1, *args, **kwargs):
    """
    Parameters
    ----------
    step_size : float, optional
      Constant scale factor of learning rate.
    friction : float, optional
      Constant scale on the friction term in the Hamiltonian system.
    """
    self.step_size = step_size
    self.friction = friction
    self.v = {z: tf.Variable(tf.zeros(qz.params.get_shape()[1:]))
              for z, qz in six.iteritems(self.latent_vars)}
    return super(SGHMC, self).initialize(*args, **kwargs)

  def build_update(self):
    """
    Simulate Hamiltonian dynamics with friction using a discretized
    integrator. Its discretization error goes to zero as the learning rate
    decreases.

    Implements the update equations from (15) of Chen et al. (2014).
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}
    old_v_sample = {z: v for z, v in six.iteritems(self.v)}

    # Simulate Hamiltonian dynamics with friction.
    friction = tf.constant(self.friction, dtype=tf.float32)
    learning_rate = tf.constant(self.step_size * 0.01, dtype=tf.float32)
    grad_log_joint = tf.gradients(self._log_joint(old_sample),
                                  list(six.itervalues(old_sample)))

    # v_sample is so named b/c it represents a velocity rather than momentum.
    sample = {}
    v_sample = {}
    for z, grad_log_p in zip(six.iterkeys(old_sample), grad_log_joint):
      qz = self.latent_vars[z]
      event_shape = qz.get_event_shape()
      normal = Normal(mu=tf.zeros(event_shape),
                      sigma=(tf.sqrt(learning_rate * friction) *
                             tf.ones(event_shape)))
      sample[z] = old_sample[z] + old_v_sample[z]
      v_sample[z] = ((1. - 0.5 * friction) * old_v_sample[z] +
                     learning_rate * grad_log_p + normal.sample())

    # Update Empirical random variables.
    assign_ops = []
    for z, qz in six.iteritems(self.latent_vars):
      variable = qz.get_variables()[0]
      assign_ops.append(tf.scatter_update(variable, self.t, sample[z]))
      assign_ops.append(tf.assign(self.v[z], v_sample[z]).op)

    # Increment n_accept.
    assign_ops.append(self.n_accept.assign_add(1))
    return tf.group(*assign_ops)

  def _log_joint(self, z_sample):
    """
    Utility function to calculate model's log joint density,
    log p(x, z), for inputs z (and fixed data x).

    Parameters
    ----------
    z_sample : dict
      Latent variable keys to samples.
    """
    if self.model_wrapper is None:
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
    else:
      x = self.data
      log_joint = self.model_wrapper.log_prob(x, z_sample)

    return log_joint
