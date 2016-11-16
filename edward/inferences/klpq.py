from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable, Normal
from edward.util import copy, log_sum_exp


class KLpq(VariationalInference):
  """Variational inference with the KL divergence

  .. math::

    KL( p(z |x) || q(z) ).

  To perform the optimization, this class uses a technique from
  adaptive importance sampling (Cappe et al., 2008).

  Notes
  -----
  KLqp also optimizes any model parameters p(z | x; \theta). It does
  this by variational EM, minimizing

  .. math::

    E_{p(z | x; \lambda)} [ \log p(x, z; \theta) ]

  with respect to \theta.

  In conditional inference, we infer z in p(z, \beta | x) while fixing
  inference over \beta using another distribution q(\beta).
  During gradient calculation, instead of using the model's density

  .. math::

    \log p(x, z^{(s)}), where z^{(s)} ~ q(z; \lambda),

  for each sample s=1,...,S, KLpq uses

  .. math::

    \log p(x, z^{(s)}, \beta^{(s)}), where
    z^{(s)} ~ q(z; \lambda) and \beta^{(s)} ~ q(beta).
  """
  def __init__(self, *args, **kwargs):
    super(KLpq, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    """
    self.n_samples = n_samples
    return super(KLpq, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    """Build loss function

    .. math::
      KL( p(z |x) || q(z) )
      =
      E_{p(z | x)} [ \log p(z | x) - \log q(z; \lambda) ]

    and stochastic gradients based on importance sampling.

    The loss function can be estimated as

    .. math::
      1/B \sum_{b=1}^B [ w_{norm}(z^b; \lambda) *
                         (\log p(x, z^b) - \log q(z^b; \lambda) ],

    where

    .. math::
      z^b \sim q(z^b; \lambda),

      w_{norm}(z^b; \lambda) = w(z^b; \lambda) / \sum_{b=1}^B (w(z^b; \lambda)),

      w(z^b; \lambda) = p(x, z^b) / q(z^b; \lambda).

    This provides a gradient,

    .. math::
      - 1/B \sum_{b=1}^B [ w_{norm}(z^b; \lambda) *
                           \partial_{\lambda} \log q(z^b; \lambda) ].
    """
    p_log_prob = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    for s in range(self.n_samples):
      scope = 'inference_' + str(id(self)) + '/' + str(s)
      z_sample = {}
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope=scope)
        z_sample[z] = qz_copy.value()
        q_log_prob[s] += tf.reduce_sum(
            qz.log_prob(tf.stop_gradient(z_sample[z])))

      if self.model_wrapper is None:
        # Form dictionary in order to replace conditioning on prior or
        # observed variable with conditioning on a specific value.
        dict_swap = z_sample
        for x, qx in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            if isinstance(qx, RandomVariable):
              qx_copy = copy(qx, scope=scope)
              dict_swap[x] = qx_copy.value()
            else:
              dict_swap[x] = qx

        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(dict_swap[z]))

        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(dict_swap[x]))
      else:
        x = self.data
        p_log_prob[s] = self.model_wrapper.log_prob(x, z_sample)

    p_log_prob = tf.pack(p_log_prob)
    q_log_prob = tf.pack(q_log_prob)

    log_w = p_log_prob - q_log_prob
    log_w_norm = log_w - log_sum_exp(log_w)
    w_norm = tf.exp(log_w_norm)

    if var_list is None:
      var_list = tf.trainable_variables()

    loss = tf.reduce_mean(w_norm * log_w)
    grads = tf.gradients(
        -tf.reduce_mean(q_log_prob * tf.stop_gradient(w_norm)),
        [v.ref() for v in var_list])
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars
