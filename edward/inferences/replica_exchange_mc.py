from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
import numpy as np

from collections import OrderedDict
from edward.inferences.monte_carlo import MonteCarlo
from edward.models import Empirical, RandomVariable
from edward.util import check_latent_vars, copy


class _stateful_lambda:
  """Class to use instead of lambda.
  lambda is affected by the change of x,
  so _stateful_lambda(x)() output x at the time of definition.
  """

  def __init__(self, x):
    self.x = x

  def __call__(self):
    return self.x


class ReplicaExchangeMC(MonteCarlo):
  """Replica Exchange MCMC [@swendsen1986replica; @hukushima1996exchange].

  #### Examples
  ```python
  cat = Categorical(probs=[0.5,0.5])
  x = Mixture(cat=cat, components=[
      MultivariateNormalDiag([0.0,0.0], [1.0,1.0]),
      MultivariateNormalDiag([10.0,10.0], [1.0,1.0])])
  proposal_x = MultivariateNormalDiag(x, [1.0,1.0])
  qx = Empirical(tf.Variable(tf.zeros([10000, 2])))
  inference = ed.ReplicaExchangeMC(latent_vars={x: qx},
                                   proposal_vars={x: proposal_x})
  ```
  """

  def __init__(self, latent_vars, proposal_vars, data=None,
               inverse_temperatures=np.logspace(0, -2, 5), exchange_freq=0.1):
    """Create an inference algorithm.

    Args:
      proposal_vars: dict of RandomVariable to RandomVariable.
        Collection of random variables to perform inference on; each is
        binded to a proposal distribution $g(z' \mid z)$.
      inverse_temperatures: list of inverse temperature.
      exchange_freq: frequency of exchanging replica.
    """
    check_latent_vars(proposal_vars)
    self.proposal_vars = proposal_vars
    super(ReplicaExchangeMC, self).__init__(latent_vars, data)

    self.n_replica = len(inverse_temperatures)
    if inverse_temperatures[0] != 1:
      raise ValueError("inverse_temperatures[0] must be 1.")
    self.inverse_temperatures = tf.cast(inverse_temperatures,
                                        dtype=list(self.latent_vars)[0].dtype)

    # Make replica.
    self.replica_vars = []
    for i in range(self.n_replica):
      self.replica_vars.append({z: Empirical(params=tf.Variable(tf.zeros(
          qz.params.shape, dtype=self.latent_vars[z].dtype))) for z, qz in
          six.iteritems(self.latent_vars)})

    self.exchange_freq = exchange_freq

  def initialize(self, *args, **kwargs):
    kwargs['auto_transform'] = False
    return super(ReplicaExchangeMC, self).initialize(*args, **kwargs)

  def build_update(self):
    """Perform sampling and exchange.
    """
    # Sample by Metropolis-Hastings for each replica.
    replica_sample = []
    replica_accept = []
    for i in range(self.n_replica):
      sample_, accept_ = self._mh_sample(self.replica_vars[i],
                                         self.inverse_temperatures[i])
      replica_sample.append(sample_)
      replica_accept.append(accept_)
    accept = replica_accept[0]

    # Variable to store order of replicas after exchange
    new_replica_idx = tf.Variable(tf.range(self.n_replica))
    new_replica_idx = tf.assign(new_replica_idx, tf.range(self.n_replica))

    # Variable to store ratio of current samples
    replica_ratio = tf.Variable(tf.zeros(
        self.n_replica, dtype=list(self.latent_vars)[0].dtype))
    replica_ratio = self._replica_ratio(replica_ratio, replica_sample)

    # Exchange adjacent replicas at frequency of exchange_freq
    u = tf.random_uniform([])
    exchange = u < self.exchange_freq
    new_replica_idx = tf.cond(
        exchange, lambda: self._replica_exchange(
            new_replica_idx, replica_ratio), lambda: new_replica_idx)

    # New replica sorted by new_replica_idx
    new_replica_sample = []
    for i in range(self.n_replica):
      new_replica_sample.append(
          {z: tf.case({tf.equal(tf.gather(new_replica_idx, i), j):
                      _stateful_lambda(replica_sample[j][z])
                      for j in range(self.n_replica)},
           default=lambda: replica_sample[0][z], exclusive=True) for z, qz in
           six.iteritems(self.latent_vars)})

    assign_ops = []

    # Update Empirical random variables.
    for z, qz in six.iteritems(self.latent_vars):
      variable = qz.get_variables()[0]
      assign_ops.append(tf.scatter_update(variable, self.t,
                                          new_replica_sample[0][z]))

    for i in range(self.n_replica):
      for z, qz in six.iteritems(self.replica_vars[i]):
        variable = qz.get_variables()[0]
        assign_ops.append(tf.scatter_update(variable, self.t,
                                            new_replica_sample[i][z]))

    # Increment n_accept (if accepted).
    assign_ops.append(self.n_accept.assign_add(tf.where(accept, 1, 0)))

    return tf.group(*assign_ops)

  def _mh_sample(self, latent_vars, inverse_temperature):
    """Draw sample by Metropolis-Hastings. Then
    accept or reject the sample based on the ratio,
    $\\text{ratio} = \\text{inverse_temperature}(
          \log p(x, z^{\\text{new}}) - \log p(x, z^{\\text{old}}) -
          \log g(z^{\\text{new}} \mid z^{\\text{old}}) +
          \log g(z^{\\text{old}} \mid z^{\\text{new}}))$
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(latent_vars)}
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

    for z in six.iterkeys(latent_vars):
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

    ratio *= inverse_temperature

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
    return sample, accept

  def _replica_exchange(self, new_replica_idx, replica_ratio):
    """Exchange replica according to the Metropolis-Hastings criterion.
    $\\text{ratio} =
          (\log p(x, z_i) - \log p(x, x_j))(\\text{inverse_temperature}_j -
          \\text{inverse_temperature}_i)
    """
    i = tf.random_uniform((), maxval=2, dtype=tf.int32)

    def cond(i, new_replica_idx):
      return tf.less(i, self.n_replica - 1)

    def body(i, new_replica_idx):
      ratio = replica_ratio[i] - replica_ratio[i + 1]
      ratio *= (self.inverse_temperatures[i + 1] - self.inverse_temperatures[i])
      u = tf.random_uniform([], dtype=ratio.dtype)
      exchange = tf.log(u) < ratio
      new_replica_idx = tf.cond(
          exchange,
          lambda: tf.scatter_update(new_replica_idx, [i, i + 1], [i + 1, i]),
          lambda: new_replica_idx)
      return [i + 2, new_replica_idx]

    return tf.while_loop(cond, body, loop_vars=[i, new_replica_idx])[1]

  def _replica_ratio(self, replica_ratio, replica_sample):
    replica_ratio = tf.assign(replica_ratio, tf.zeros(
        self.n_replica, dtype=list(self.latent_vars)[0].dtype))

    dict_swap = {}
    for x, qx in six.iteritems(self.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope='conditional')
          dict_swap[x] = qx_copy.value()
        else:
          dict_swap[x] = qx

    for i in range(self.n_replica):
      dict_swap_i = dict_swap.copy()
      dict_swap_i.update(replica_sample[i])

      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      scope_i = base_scope + '_%d' % i

      for z in six.iterkeys(self.latent_vars):
        # Build priors p(z_i) and p(z_j).
        z_i = copy(z, dict_swap_i, scope=scope_i)
        # Increment ratio.
        replica_ratio = tf.scatter_update(
            replica_ratio, i,
            replica_ratio[i] + tf.reduce_sum(z_i.log_prob(dict_swap_i[z])))

      for x in six.iterkeys(self.data):
        if isinstance(x, RandomVariable):
          # Build likelihoods p(x | z_i) and p(x | z_j).
          x_z_i = copy(x, dict_swap_i, scope=scope_i)
          # Increment ratio.
          replica_ratio = tf.scatter_update(
              replica_ratio, i,
              replica_ratio[i] + tf.reduce_sum(x_z_i.log_prob(dict_swap[x])))
    return replica_ratio
