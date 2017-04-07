from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
import numpy as np

from edward.inferences.monte_carlo import MonteCarlo
from edward.models import RandomVariable
from edward.util import copy

try:
  from edward.models import Normal
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class mSGNHT(MonteCarlo):
  """multivariate Stochastic Gradient Nose-Hoover Thermostats with Symmetric Splitting Integrator(Chen et al. 2015).

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
    >>> z = Normal(mu=0.0, sigma=1.0)
    >>> x = Normal(mu=tf.ones(10) * z, sigma=1.0)
    >>>
    >>> qz = Empirical(tf.Variable(tf.zeros(500)))
    >>> data = {x: np.array([0.0] * 10, dtype=np.float32)}
    >>> inference = ed.SGLD({z: qz}, data)
    """
    super(mSGNHT, self).__init__(*args, **kwargs)

  def initialize(self, D=0.75, step_size=0.0001, *args, **kwargs):
    """
    Parameters
    ----------
    D : positive float, optional
      A positive constant of equation (2) in Chen, et al.(2015).
    step_size : float, optional
      Constant scale factor of learning rate.
    """
    self.D = float(D)
    self.h = float(step_size)
    self.p = {z: tf.Variable(tf.zeros(qz.params.shape[1:]))
              for z, qz in six.iteritems(self.latent_vars)}
    self.xi = self.p
    return super(mSGNHT, self).initialize(*args, **kwargs)

  def build_update(self):
    """Simulate mSGNHT using a symmetric splitting integrator(SSI).
     SSI is more accurate than the the traditional Euler integrator.

    Notes
    -----
    The updates assume each Empirical random variable is directly
    parameterized by ``tf.Variable``s.
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}

    # Simulate mSGNHT according to equation (3) in Chen, et al.(2015).
    first_sample = {}
    first_p_sample = {}
    first_xi_sample = {}

    second_p_sample = {}

    final_sample = {}
    final_p_sample = {}
    final_xi_sample = {}

    for z in six.iterkeys(self.latent_vars):
      # Simulate "A" step of the integrator
      first_sample[z] = old_sample[z] + self.p[z] * 0.5 * self.h
      first_xi_sample[z] = self.xi[z] + (tf.multiply(self.p[z], self.p[z]) - 1) * self.h * 0.5

      # Simulate "B" step of the integrator
      first_p_sample[z] = tf.multiply(tf.exp(- first_xi_sample[z] * self.h * 0.5), self.p[z])

    grad_log_joint = tf.gradients(self._log_joint(first_sample),
                                  list(six.itervalues(first_sample)))
    for z, grad_log_p in zip(six.iterkeys(first_sample), grad_log_joint):
      # Simulate "O" step of the integrator
      qz = self.latent_vars[z]
      zeta = tf.contrib.distributions.Normal(mu=0., sigma=self.h)
      second_p_sample[z] = first_p_sample[z] - grad_log_p * self.h + \
                           tf.sqrt(2 * self.D) * zeta.sample(sample_shape=qz.get_event_shape())

      # Simulate "B" step of the integrator
      final_p_sample[z] = tf.multiply(tf.exp(- first_xi_sample[z] * self.h * 0.5), second_p_sample[z])
      # Simulate "A" step of the integrator
      final_sample[z] = first_sample[z] + final_p_sample[z] * 0.5 * self.h
      final_xi_sample[z] = first_xi_sample[z] + (tf.multiply(final_p_sample[z], final_p_sample[z]) - 1) * self.h * 0.5

    # Update Empirical random variables.
    assign_ops = []
    for z, qz in six.iteritems(self.latent_vars):
      variable = qz.get_variables()[0]
      assign_ops.append(tf.assign(self.p[z], final_p_sample[z]).op)
      assign_ops.append(tf.scatter_update(variable, self.t, final_sample[z]))
      assign_ops.append(tf.assign(self.xi[z], final_xi_sample[z]).op)

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
