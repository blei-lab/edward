from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from collections import OrderedDict
from edward.inferences.monte_carlo import MonteCarlo
from edward.models import RandomVariable
from edward.util import copy

try:
  from edward.models import Normal, Uniform
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class HMC(MonteCarlo):
  """Hamiltonian Monte Carlo, also known as hybrid Monte Carlo
  [@duane1987hybrid; @neal2011mcmc].

  #### Notes

  In conditional inference, we infer $z$ in $p(z, \\beta
  \mid x)$ while fixing inference over $\\beta$ using another
  distribution $q(\\beta)$.
  `HMC` substitutes the model's log marginal density

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
  inference = ed.HMC({mu: qmu}, {x: np.zeros(10, dtype=np.float32)})
  ```
  """
  def __init__(self, *args, **kwargs):
    super(HMC, self).__init__(*args, **kwargs)

  def initialize(self, step_size=0.25, n_steps=2, *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.

    Args:
      step_size: float, optional.
        Step size of numerical integrator.
      n_steps: int, optional.
        Number of steps of numerical integrator.
    """
    self.step_size = step_size
    self.n_steps = n_steps
    # store global scope for log joint calculations
    self._scope = tf.get_default_graph().unique_name("inference") + '/'
    return super(HMC, self).initialize(*args, **kwargs)

  def build_update(self):
    """Simulate Hamiltonian dynamics using a numerical integrator.
    Correct for the integrator's discretization error using an
    acceptance ratio.

    #### Notes

    The updates assume each Empirical random variable is directly
    parameterized by `tf.Variable`s.
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}
    old_sample = OrderedDict(old_sample)

    # Sample momentum.
    old_r_sample = OrderedDict()
    for z, qz in six.iteritems(self.latent_vars):
      event_shape = qz.event_shape
      normal = Normal(loc=tf.zeros(event_shape, dtype=qz.dtype),
                      scale=tf.ones(event_shape, dtype=qz.dtype))
      old_r_sample[z] = normal.sample()

    # Simulate Hamiltonian dynamics.
    new_sample, new_r_sample = leapfrog(old_sample, old_r_sample,
                                        self.step_size, self._log_joint,
                                        self.n_steps)

    # Calculate acceptance ratio.
    ratio = tf.reduce_sum([0.5 * tf.reduce_sum(tf.square(r))
                           for r in six.itervalues(old_r_sample)])
    ratio -= tf.reduce_sum([0.5 * tf.reduce_sum(tf.square(r))
                            for r in six.itervalues(new_r_sample)])
    ratio += self._log_joint(new_sample)
    ratio -= self._log_joint(old_sample)

    # Accept or reject sample.
    u = Uniform(low=tf.constant(0.0, dtype=ratio.dtype),
                high=tf.constant(1.0, dtype=ratio.dtype)).sample()
    accept = tf.log(u) < ratio
    sample_values = tf.cond(accept, lambda: list(six.itervalues(new_sample)),
                            lambda: list(six.itervalues(old_sample)))
    if not isinstance(sample_values, list):
      # `tf.cond` returns tf.Tensor if output is a list of size 1.
      sample_values = [sample_values]

    sample = {z: sample_value for z, sample_value in
              zip(six.iterkeys(new_sample), sample_values)}

    # Update Empirical random variables.
    assign_ops = []
    for z, qz in six.iteritems(self.latent_vars):
      variable = qz.get_variables()[0]
      assign_ops.append(tf.scatter_update(variable, self.t, sample[z]))

    # Increment n_accept (if accepted).
    assign_ops.append(self.n_accept.assign_add(tf.where(accept, 1, 0)))
    return tf.group(*assign_ops)

  def _log_joint(self, z_sample):
    """Utility function to calculate model's log joint density,
    log p(x, z), for inputs z (and fixed data x).

    Args:
      z_sample: dict.
        Latent variable keys to samples.
    """
    scope = self._scope + tf.get_default_graph().unique_name("sample")
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
      log_joint += tf.reduce_sum(z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(self.data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        log_joint += tf.reduce_sum(x_copy.log_prob(dict_swap[x]))

    return log_joint


def leapfrog(z_old, r_old, step_size, log_joint, n_steps):
  z_new = z_old.copy()
  r_new = r_old.copy()

  grad_log_joint = tf.gradients(log_joint(z_new), list(six.itervalues(z_new)))
  for _ in range(n_steps):
    for i, key in enumerate(six.iterkeys(z_new)):
      z, r = z_new[key], r_new[key]
      r_new[key] = r + 0.5 * step_size * tf.convert_to_tensor(grad_log_joint[i])
      z_new[key] = z + step_size * r_new[key]

    grad_log_joint = tf.gradients(log_joint(z_new), list(six.itervalues(z_new)))
    for i, key in enumerate(six.iterkeys(z_new)):
      r_new[key] += 0.5 * step_size * tf.convert_to_tensor(grad_log_joint[i])

  return z_new, r_new
