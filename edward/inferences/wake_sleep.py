from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable
from edward.util import copy, get_descendants


class WakeSleep(VariationalInference):
  """Wake-Sleep algorithm [@hinton1995wake].

  Given a probability model $p(x, z; \\theta)$ and variational
  distribution $q(z\mid x; \\lambda)$, wake-sleep alternates between
  two phases:

  + In the wake phase, $\log p(x, z; \\theta)$ is maximized with
  respect to model parameters $\\theta$ using bottom-up samples
  $z\sim q(z\mid x; \lambda)$.
  + In the sleep phase, $\log q(z\mid x; \lambda)$ is maximized with
  respect to variational parameters $\lambda$ using top-down
  "fantasy" samples $z\sim p(x, z; \\theta)$.

  @hinton1995wake justify wake-sleep under the variational lower
  bound of the description length,

  $\mathbb{E}_{q(z\mid x; \lambda)} [
      \log p(x, z; \\theta) - \log q(z\mid x; \lambda)].$

  Maximizing it with respect to $\\theta$ corresponds to the wake phase.
  Instead of maximizing it with respect to $\lambda$ (which
  corresponds to minimizing $\\text{KL}(q\|p)$), the sleep phase
  corresponds to minimizing the reverse KL $\\text{KL}(p\|q)$ in
  expectation over the data distribution.

  #### Notes

  In conditional inference, we infer $z$ in $p(z, \\beta
  \mid x)$ while fixing inference over $\\beta$ using another
  distribution $q(\\beta)$. During gradient calculation, instead
  of using the model's density

  $\log p(x, z^{(s)}), z^{(s)} \sim q(z; \lambda),$

  for each sample $s=1,\ldots,S$, `WakeSleep` uses

  $\log p(x, z^{(s)}, \\beta^{(s)}),$

  where $z^{(s)} \sim q(z; \lambda)$ and $\\beta^{(s)}
  \sim q(\\beta)$.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  def __init__(self, *args, **kwargs):
    super(WakeSleep, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, phase_q='sleep', *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.

    Args:
      n_samples: int.
        Number of samples for calculating stochastic gradients during
        wake and sleep phases.
      phase_q: str.
        Phase for updating parameters of q. If 'sleep', update using
        a sample from p. If 'wake', update using a sample from q.
        (Unlike reparameterization gradients, the sample is held
        fixed.)
    """
    self.n_samples = n_samples
    self.phase_q = phase_q
    return super(WakeSleep, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    p_log_prob = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    base_scope = tf.get_default_graph().unique_name("inference") + '/'
    for s in range(self.n_samples):
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      scope = base_scope + tf.get_default_graph().unique_name("q_sample")
      dict_swap = {}
      for x, qx in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      # Sample z ~ q(z), then compute log p(x, z).
      q_dict_swap = dict_swap.copy()
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope=scope)
        q_dict_swap[z] = qz_copy.value()
        if self.phase_q != 'sleep':
          # If not sleep phase, compute log q(z).
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) *
              qz_copy.log_prob(tf.stop_gradient(q_dict_swap[z])))

      for z in six.iterkeys(self.latent_vars):
        z_copy = copy(z, q_dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            self.scale.get(z, 1.0) * z_copy.log_prob(q_dict_swap[z]))

      for x in six.iterkeys(self.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, q_dict_swap, scope=scope)
          p_log_prob[s] += tf.reduce_sum(
              self.scale.get(x, 1.0) * x_copy.log_prob(q_dict_swap[x]))

      if self.phase_q == 'sleep':
        # Sample z ~ p(z), then compute log q(z).
        scope = base_scope + tf.get_default_graph().unique_name("p_sample")
        p_dict_swap = dict_swap.copy()
        for z, qz in six.iteritems(self.latent_vars):
          # Copy p(z) to obtain new set of prior samples.
          z_copy = copy(z, scope=scope)
          p_dict_swap[qz] = z_copy.value()
        for qz in six.itervalues(self.latent_vars):
          qz_copy = copy(qz, p_dict_swap, scope=scope)
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) *
              qz_copy.log_prob(tf.stop_gradient(p_dict_swap[qz])))

    p_log_prob = tf.reduce_mean(p_log_prob)
    q_log_prob = tf.reduce_mean(q_log_prob)
    reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())

    if self.logging:
      tf.summary.scalar("loss/p_log_prob", p_log_prob,
                        collections=[self._summary_key])
      tf.summary.scalar("loss/q_log_prob", q_log_prob,
                        collections=[self._summary_key])
      tf.summary.scalar("loss/reg_penalty", reg_penalty,
                        collections=[self._summary_key])

    loss_p = -p_log_prob + reg_penalty
    loss_q = -q_log_prob + reg_penalty

    q_rvs = list(six.itervalues(self.latent_vars))
    q_vars = [v for v in var_list
              if len(get_descendants(tf.convert_to_tensor(v), q_rvs)) != 0]
    q_grads = tf.gradients(loss_q, q_vars)
    p_vars = [v for v in var_list if v not in q_vars]
    p_grads = tf.gradients(loss_p, p_vars)
    grads_and_vars = list(zip(q_grads, q_vars)) + list(zip(p_grads, p_vars))
    return loss_p, grads_and_vars
