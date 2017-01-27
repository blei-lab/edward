from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from collections import OrderedDict
from edward.inferences.monte_carlo import MonteCarlo
from edward.models import RandomVariable, Uniform
from edward.util import copy


class MetropolisHastings(MonteCarlo):
  """Metropolis-Hastings.

  Notes
  -----
  In conditional inference, we infer :math:`z` in :math:`p(z, \\beta
  \mid x)` while fixing inference over :math:`\\beta` using another
  distribution :math:`q(\\beta)`.
  To calculate the acceptance ratio, ``MetropolisHastings`` uses an
  estimate of the marginal density,

  .. math::

    p(x, z) = \mathbb{E}_{q(\\beta)} [ p(x, z, \\beta) ]
            \\approx p(x, z, \\beta^*)

  leveraging a single Monte Carlo sample, where :math:`\\beta^* \sim
  q(\\beta)`. This is unbiased (and therefore asymptotically exact as a
  pseudo-marginal method) if :math:`q(\\beta) = p(\\beta \mid x)`.
  """
  def __init__(self, latent_vars, proposal_vars, data=None, model_wrapper=None):
    """
    Parameters
    ----------
    proposal_vars : dict of RandomVariable to RandomVariable
      Collection of random variables to perform inference on; each is
      binded to a proposal distribution :math:`g(z' \mid z)`.

    Examples
    --------
    >>> z = Normal(mu=0.0, sigma=1.0)
    >>> x = Normal(mu=tf.ones(10) * z, sigma=1.0)
    >>>
    >>> qz = Empirical(tf.Variable(tf.zeros([500])))
    >>> proposal_z = Normal(mu=z, sigma=0.5)
    >>> data = {x: np.array([0.0] * 10, dtype=np.float32)}
    >>> inference = ed.MetropolisHastings({z: qz}, {z: proposal_z}, data)
    """
    self.proposal_vars = proposal_vars
    super(MetropolisHastings, self).__init__(latent_vars, data, model_wrapper)

  def build_update(self):
    """
    Draw sample from proposal conditional on last sample. Then accept
    or reject the sample based on the ratio,

    .. math::
      \\text{ratio} = \log p(x, z^{new}) - \log p(x, z^{old}) +
        \log g(z^{new} \mid z^{old}) - \log g(z^{old} \mid z^{new})

    Notes
    -----
    The updates assume each Empirical random variable is directly
    parameterized by tf.Variables().
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
    scope_old = 'inference_' + str(id(self)) + '/old'
    scope_new = 'inference_' + str(id(self)) + '/new'

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

    for z, proposal_z in six.iteritems(self.proposal_vars):
      # Build proposal g(zold | znew).
      proposal_zold = copy(proposal_z, dict_swap_new, scope=scope_new)
      # Increment ratio.
      ratio -= tf.reduce_sum(proposal_zold.log_prob(dict_swap_old[z]))

    if self.model_wrapper is None:
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
    else:
        x = self.data
        ratio += self.model_wrapper.log_prob(x, new_sample)
        ratio -= self.model_wrapper.log_prob(x, old_sample)

    # Accept or reject sample.
    u = Uniform().sample()
    accept = tf.log(u) < ratio
    sample_values = tf.cond(accept, lambda: list(six.itervalues(new_sample)),
                            lambda: list(six.itervalues(old_sample)))
    if not isinstance(sample_values, list):
      # ``tf.cond`` returns tf.Tensor if output is a list of size 1.
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
