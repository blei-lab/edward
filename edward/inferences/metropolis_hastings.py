from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from collections import OrderedDict
from edward.inferences.monte_carlo import MonteCarlo
from edward.models import RandomVariable
from edward.util import check_latent_vars, copy


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
    return super(MetropolisHastings, self).initialize(*args, **kwargs)

  def build_update(self):
    """Draw sample from proposal conditional on last sample. Then
    accept or reject the sample based on the ratio,

    $\\text{ratio} =
          \log p(x, z^{\\text{new}}) - \log p(x, z^{\\text{old}}) -
          \log g(z^{\\text{new}} \mid z^{\\text{old}}) +
          \log g(z^{\\text{old}} \mid z^{\\text{new}})$

    #### Notes

    The updates assume each Empirical random variable is directly
    parameterized by `tf.Variable`s.
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}
    old_sample = OrderedDict(old_sample)

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
    scope_new = base_scope + 'new'

    # Draw proposed sample and calculate acceptance ratio.
    new_sample = old_sample.copy()  # copy to ensure same order
    ratio = 0.0
    for z, proposal_z in six.iteritems(self.proposal_vars):
      # Build proposal g(znew | zold).
      proposal_znew = copy(proposal_z, dict_swap_old, scope=scope_old)
      # Sample znew ~ g(znew | zold).
      new_sample[z] = proposal_znew.value()
      # Increment ratio.
      ratio -= tf.reduce_sum(proposal_znew.log_prob(new_sample[z]))

    dict_swap_new = dict_swap.copy()
    dict_swap_new.update(new_sample)

    for z, proposal_z in six.iteritems(self.proposal_vars):
      # Build proposal g(zold | znew).
      proposal_zold = copy(proposal_z, dict_swap_new, scope=scope_new)
      # Increment ratio.
      ratio += tf.reduce_sum(proposal_zold.log_prob(dict_swap_old[z]))

    for z in six.iterkeys(self.latent_vars):
      # Build priors p(znew) and p(zold).
      znew = copy(z, dict_swap_new, scope=scope_new)
      zold = copy(z, dict_swap_old, scope=scope_old)
      # Increment ratio.
      ratio += tf.reduce_sum(znew.log_prob(dict_swap_new[z]))
      ratio -= tf.reduce_sum(zold.log_prob(dict_swap_old[z]))

    for x in six.iterkeys(self.data):
      if isinstance(x, RandomVariable):
        # Build likelihoods p(x | znew) and p(x | zold).
        x_znew = copy(x, dict_swap_new, scope=scope_new)
        x_zold = copy(x, dict_swap_old, scope=scope_old)
        # Increment ratio.
        ratio += tf.reduce_sum(x_znew.log_prob(dict_swap[x]))
        ratio -= tf.reduce_sum(x_zold.log_prob(dict_swap[x]))

    # Accept or reject sample.
    u = tf.random_uniform([], dtype=ratio.dtype)
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
