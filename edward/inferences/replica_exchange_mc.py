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


class memory_lambda:
    def __init__(self, x):
        self.x = x
    def __call__(self):
        return self.x
    
class ReplicaExchangeMC(MonteCarlo):
  """
  Replica Exchange MCMC(https://en.wikipedia.org/wiki/Parallel_tempering)
  
  #### Examples
  ```python
  cat = Categorical(probs=[0.5,0.5])
  x = Mixture(cat=cat, components=
                [MultivariateNormalDiag([0.0,0.0], [1.0,1.0]),
                 MultivariateNormalDiag([10.0,10.0], [1.0,1.0])])
  proposal_x = MultivariateNormalDiag(x, [1.0,1.0])
  qx = Empirical(tf.Variable(tf.zeros([10000, 2]))) #初期値
  inference = ed.ReplicaExchangeMC(latent_vars={x: qx},
                                 proposal_vars={x: proposal_x})
  ```
  """
  def __init__(self, latent_vars, proposal_vars, data=None, betas=np.logspace(0,-1, 10)):
    """Create an inference algorithm.
    Args:
      proposal_vars: dict of RandomVariable to RandomVariable.
        Collection of random variables to perform inference on; each is
        binded to a proposal distribution $g(z' \mid z)$.
      betas: list of inverse temperature.
    """
    check_latent_vars(proposal_vars)
    self.proposal_vars = proposal_vars
    
    self.betas = betas.astype(np.float32)
    if self.betas[0] != 1:
      raise ValueError("betas[0] must be 1.")

    self.replica_vars = []
    for beta in self.betas:
        self.replica_vars.append({z: Empirical(params=tf.Variable(tf.zeros(
                qz.params.shape))) for z, qz in six.iteritems(latent_vars)})
    super(ReplicaExchangeMC, self).__init__(latent_vars, data)
  def initialize(self, *args, **kwargs):
    kwargs['auto_transform'] = False
    return super(ReplicaExchangeMC, self).initialize(*args, **kwargs)
  def build_update(self):
    """
    Perform sampling and exchange.
    """
    replica_sample = []
    replica_accept = []
    for i, beta in enumerate(self.betas):
        sample_, accept_ = self.mh_sample(self.replica_vars[i], beta)
        replica_sample.append(sample_)
        replica_accept.append(accept_)
    accept = replica_accept[0]
    
    new_replica_idx = tf.Variable(tf.range(len(self.replica_vars)), name='new_replica_idx')
    new_replica_idx = tf.scatter_update(new_replica_idx, tf.range(len(self.replica_vars)), tf.range(len(self.replica_vars)))
    
    cand = tf.gather(tf.random_shuffle(tf.range(len(self.replica_vars))),[0,1])
    exchange = self.replica_exchange(cand, replica_sample)
    
    new_replica_idx = tf.cond(exchange, 
                              lambda: tf.scatter_update(new_replica_idx, cand, cand[::-1]),
                              lambda: new_replica_idx)
    
    accept = tf.logical_or(accept,tf.not_equal(new_replica_idx[0],0))
    new_replica_sample = []
    for i in range(len(self.betas)):
        new_replica_sample.append(tf.case({tf.equal(tf.gather(new_replica_idx,i),j):
            memory_lambda(replica_sample[j]) for j in range(len(self.betas))},
             exclusive=True))
    
    # Update Empirical random variables.
    assign_ops = []
    for z, qz in six.iteritems(self.latent_vars):
      variable = qz.get_variables()[0]
      assign_ops.append(tf.scatter_update(variable, self.t, new_replica_sample[0][z]))
        
    # Increment n_accept (if accepted).
    assign_ops.append(self.n_accept.assign_add(tf.where(accept, 1, 0)))
    
    for i, beta in enumerate(self.betas):
        for z, qz in six.iteritems(self.replica_vars[i]):
            variable = qz.get_variables()[0]
            #assign_ops.append(tf.scatter_update(variable, self.t, replica_sample[i][z]))
            assign_ops.append(tf.scatter_update(variable, self.t, new_replica_sample[i][z]))


    return tf.group(*assign_ops)
  def mh_sample(self, latent_vars, beta):
    """Draw sample by Metropolis-Hastings. Then
    accept or reject the sample based on the ratio,
    $\\text{ratio} =
          \beta\log p(x, z^{\\text{new}}) - \beta\log p(x, z^{\\text{old}}) -
          \beta\log g(z^{\\text{new}} \mid z^{\\text{old}}) +
          \beta\log g(z^{\\text{old}} \mid z^{\\text{new}})$
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

    ratio *= beta
    
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
  def replica_exchange(self, candidate, replica_sample):
    """
    Exchange replica according to the Metropolis-Hastings criterion.
    Args:
        candidate: indices of replica that may be exchanged
        replica_sample: list of replica
    """
    sample_i = tf.case({tf.equal(candidate[0],i):memory_lambda(replica_sample[i])for i
                        in range(len(self.betas))}, exclusive=True)
    beta_i = tf.case({tf.equal(candidate[0],i):memory_lambda(beta)for i, beta
                        in enumerate(self.betas)}, exclusive=True)
    sample_j = tf.case({tf.equal(candidate[1],i):memory_lambda(replica_sample[i])for i
                        in range(len(self.betas))}, exclusive=True)
    beta_j = tf.case({tf.equal(candidate[1],i):memory_lambda(beta)for i, beta
                        in enumerate(self.betas)}, exclusive=True)
    
    self.sample_i = sample_i
    self.sample_j = sample_j
    
    ratio = 0.0
    
    dict_swap = {}
    for x, qx in six.iteritems(self.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope='conditional')
          dict_swap[x] = qx_copy.value()
        else:
          dict_swap[x] = qx
    dict_swap_i = dict_swap.copy()
    dict_swap_i.update(sample_i)
    dict_swap_j = dict_swap.copy()
    dict_swap_j.update(sample_j)
          
    base_scope = tf.get_default_graph().unique_name("inference") + '/'
    scope_i = base_scope + '_i'
    scope_j = base_scope + '_j'
    
    for z in six.iterkeys(self.latent_vars):
      z_i = copy(z, dict_swap_i, scope=scope_i)
      z_j = copy(z, dict_swap_j, scope=scope_j)
      ratio += tf.reduce_sum(z_i.log_prob(dict_swap_i[z]))
      ratio -= tf.reduce_sum(z_j.log_prob(dict_swap_j[z]))

    for x in six.iterkeys(self.data):
      if isinstance(x, RandomVariable):
        x_z_i = copy(x, dict_swap_i, scope=scope_i)
        x_z_j = copy(x, dict_swap_j, scope=scope_j)
        
        # Increment ratio.
        ratio += tf.reduce_sum(x_z_i.log_prob(dict_swap[x]))
        ratio -= tf.reduce_sum(x_z_j.log_prob(dict_swap[x]))
    
    ratio *= beta_j - beta_i
    
    u = tf.random_uniform([], dtype=ratio.dtype)
    exchange_accept = tf.log(u) < ratio    
    return exchange_accept