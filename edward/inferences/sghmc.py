from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.monte_carlo import MonteCarlo
from edward.models import RandomVariable, Empirical
from edward.util import copy


class SGHMC(MonteCarlo):
  """Stochastic gradient Hamiltonian Monte Carlo [@chen2014stochastic].

  #### Notes

  In conditional inference, we infer $z$ in $p(z, \\beta
  \mid x)$ while fixing inference over $\\beta$ using another
  distribution $q(\\beta)$.
  `SGHMC` substitutes the model's log marginal density

  $\log p(x, z) = \log \mathbb{E}_{q(\\beta)} [ p(x, z, \\beta) ]
                \\approx \log p(x, z, \\beta^*)$

  leveraging a single Monte Carlo sample, where $\\beta^* \sim
  q(\\beta)$. This is unbiased (and therefore asymptotically exact as a
  pseudo-marginal method) if $q(\\beta) = p(\\beta \mid x)$.

  #### Examples

  ```python
  mu = Normal(loc=0.0, scale=1.0)
  x = Normal(loc=mu, scale=1.0, sample_shape=10)

  qmu = Empirical(tf.Variable(tf.zeros(500)))
  inference = ed.SGHMC({mu: qmu}, {x: np.zeros(10, dtype=np.float32)})
  ```
  """
  def __init__(self, *args, **kwargs):
    super(SGHMC, self).__init__(*args, **kwargs)

  def initialize(self, step_size=0.25, friction=0.1, *args, **kwargs):
    """Initialize inference algorithm.

    Args:
      step_size: float.
        Constant scale factor of learning rate.
      friction: float.
        Constant scale on the friction term in the Hamiltonian system.
    """
    self.step_size = step_size
    self.friction = friction
    self.v = {z: tf.Variable(tf.zeros(qz.params.shape[1:], dtype=qz.dtype))
              for z, qz in six.iteritems(self.latent_vars)}
    return super(SGHMC, self).initialize(*args, **kwargs)

  def build_update(self):
    """Simulate Hamiltonian dynamics with friction using a discretized
    integrator. Its discretization error goes to zero as the learning
    rate decreases.

    Implements the update equations from (15) of @chen2014stochastic.
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}
    old_v_sample = {z: v for z, v in six.iteritems(self.v)}

    # Simulate Hamiltonian dynamics with friction.
    learning_rate = self.step_size * 0.01
    grad_log_joint = tf.gradients(self._log_joint(old_sample),
                                  list(six.itervalues(old_sample)))

    # v_sample is so named b/c it represents a velocity rather than momentum.
    sample = {}
    v_sample = {}
    for z, grad_log_p in zip(six.iterkeys(old_sample), grad_log_joint):
      qz = self.latent_vars[z]
      event_shape = qz.event_shape
      stddev = tf.sqrt(tf.cast(learning_rate * self.friction, qz.dtype))
      normal = tf.random_normal(event_shape, dtype=qz.dtype)
      sample[z] = old_sample[z] + old_v_sample[z]
      v_sample[z] = ((1.0 - 0.5 * self.friction) * old_v_sample[z] +
                     learning_rate * tf.convert_to_tensor(grad_log_p) +
                     stddev * normal)

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
    """Utility function to calculate model's log joint density,
    log p(x, z), for inputs z (and fixed data x).

    Args:
      z_sample: dict.
        Latent variable keys to samples.
    """
    scope = tf.get_default_graph().unique_name("inference")
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
