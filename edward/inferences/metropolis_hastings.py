from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from collections import OrderedDict
from edward.inferences.monte_carlo import MonteCarlo
from edward.models import RandomVariable
from edward.util import check_latent_vars, copy

try:
  from edward.models import Uniform
  from tensorflow.contrib.bayesflow.metropolis_hastings import evolve
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class MetropolisHastings(MonteCarlo):
  """Metropolis-Hastings [@metropolis1953equation; @hastings1970monte].

  #### Notes

  In conditional inference, we infer $z$ in $p(z, \\beta
  \mid x)$ while fixing inference over $\\beta$ using another
  distribution $q(\\beta)$.
  To calculate the acceptance ratio, `MetropolisHastings` uses an
  estimate of the marginal density,

  $p(x, z) = \mathbb{E}_{q(\\beta)} [ p(x, z, \\beta) ]
            \\approx p(x, z, \\beta^*)$

  leveraging a single Monte Carlo sample, where $\\beta^* \sim
  q(\\beta)$. This is unbiased (and therefore asymptotically exact as a
  pseudo-marginal method) if $q(\\beta) = p(\\beta \mid x)$.

  `MetropolisHastings` assumes the proposal distribution has the same
  support as the prior. The `auto_transform` attribute in
  the method `initialize()` is not applicable.

  #### Examples

  ```python
  mu = Normal(loc=0.0, scale=1.0)
  x = Normal(loc=mu, scale=1.0, sample_shape=10)

  qmu = Empirical(tf.Variable(tf.zeros(500)))
  proposal_mu = Normal(loc=mu, scale=0.5)
  inference = ed.MetropolisHastings({mu: qmu}, {mu: proposal_mu},
                                    data={x: np.zeros(10, dtype=np.float32)})
  ```
  """
  def __init__(self, latent_vars, proposal_vars, data=None):
    """Create an inference algorithm.

    Args:
      proposal_vars: dict of RandomVariable to RandomVariable.
        Collection of random variables to perform inference on; each is
        binded to a proposal distribution $g(z' \mid z)$.
    """
    check_latent_vars(proposal_vars)
    self.proposal_vars = proposal_vars
    super(MetropolisHastings, self).__init__(latent_vars, data)

  def initialize(self, *args, **kwargs):
    kwargs['auto_transform'] = False

    # TODO In general, each latent variable has arbitrary shape and
    # dtype. We cannot simply batch them into a single tf.Tensor with
    # an extra dimension. How do we handle this with ``evolve``?
    initial_sample = tf.stack([tf.gather(qz.params, 0)
                               for qz in six.itervalues(self.latent_vars)])
    self._state = tf.Variable(initial_sample, trainable=False, name="state")
    self._state_log_density = tf.Variable(
        self._log_joint(initial_sample),
        trainable=False, name="state_log_density")
    self._log_accept_ratio = tf.Variable(
        tf.zeros_like(self._state_log_density.initialized_value()),
        trainable=False, name="log_accept_ratio")
    return super(MetropolisHastings, self).initialize(*args, **kwargs)

  def build_update(self):
    """Draw sample from proposal conditional on last sample. Then
    accept or reject the sample based on the ratio,

    $\\text{ratio} =
          \log p(x, z^{\\text{new}}) - \log p(x, z^{\\text{old}}) +
          \log g(z^{\\text{new}} \mid z^{\\text{old}}) -
          \log g(z^{\\text{old}} \mid z^{\\text{new}})$

    #### Notes

    The updates assume each Empirical random variable is directly
    parameterized by `tf.Variable`s.
    """
    old_state = self._state
    forward_step = evolve(self._state,
                          self._state_log_density,
                          self._log_accept_ratio,
                          self._log_density,
                          self._proposal_fn,
                          n_steps=1)
    assign_ops = [forward_step]

    with tf.control_dependencies([forward_step]):
      # Update Empirical random variables.
      for state, qz in zip(tf.unstack(self._state),
                           six.itervalues(self.latent_vars)):
        variable = qz.get_variables()[0]
        assign_ops.append(tf.scatter_update(variable, self.t, state))

      # Increment n_accept (if accepted).
      # TODO old_state might always be same. It would be great if we
      # could more naturally get the acceptance rate from ``evolve``.
      is_proposal_accepted = tf.where(
          tf.reduce_any(tf.not_equal(old_state, self._state)), 1, 0)
      assign_ops.append(self.n_accept.assign_add(is_proposal_accepted))

    return tf.group(*assign_ops)

  def _log_joint(self, state):
    """Utility function to calculate model's log joint density,
    log p(x, z), for inputs z (and fixed data x).

    Args:
      state: tf.Tensor.
    """
    scope = self._scope + tf.get_default_graph().unique_name("sample")
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    # TODO verify ordering is preserved
    dict_swap = {z: sample for z, sample in
                 zip(six.iterkeys(self.latent_vars), state)}
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

  def proposal_fn(state):
    """Utility function to propose new state,
    znew ~ g(znew | zold) for inputs zold, and return the log density
    ratio of log g(znew | zold) - log g(zold | znew).

    Args:
      state: tf.Tensor.
    """
    # TODO verify ordering is preserved
    old_sample = {z: sample for z, sample in
                  zip(six.iterkeys(self.latent_vars), state)}

    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    dict_swap = {}
    for x, qx in six.iteritems(self.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope='conditional')
          dict_swap[x] = qx_copy.value()
        else:
          dict_swap[x] = qx

    dict_swap_old = dict_swap.copy()
    dict_swap_old.update(old_sample)
    base_scope = tf.get_default_graph().unique_name("inference") + '/'
    scope_old = base_scope + 'old'

    # Draw proposed sample and calculate acceptance ratio.
    new_sample = old_sample.copy()  # copy to ensure same order
    ratio = 0.0
    for z, proposal_z in six.iteritems(self.proposal_vars):
      # Build proposal g(znew | zold).
      proposal_znew = copy(proposal_z, dict_swap_old, scope=scope_old)
      # Sample znew ~ g(znew | zold).
      new_sample[z] = proposal_znew.value()
      # Increment ratio.
      ratio += tf.reduce_sum(proposal_znew.log_prob(new_sample[z]))

    dict_swap_new = dict_swap.copy()
    dict_swap_new.update(new_sample)
    scope_new = base_scope + 'new'

    for z, proposal_z in six.iteritems(self.proposal_vars):
      # Build proposal g(zold | znew).
      proposal_zold = copy(proposal_z, dict_swap_new, scope=scope_new)
      # Increment ratio.
      ratio -= tf.reduce_sum(proposal_zold.log_prob(dict_swap_old[z]))

    # TODO verify ordering is preserved
    new_sample = tf.stack(list(six.itervalues(new_sample)))
    return (new_sample, ratio)
