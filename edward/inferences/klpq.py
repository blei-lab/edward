from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable, Normal
from edward.util import copy, log_sum_exp


class KLpq(VariationalInference):
  """A variational inference method that minimizes the Kullback-Leibler
  divergence from the posterior to the variational model (Cappe et al., 2008)

  .. math::

    KL( p(z |x) || q(z) ).
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

  def build_loss(self):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    .. math::
      KL( p(z |x) || q(z) )
      =
      E_{p(z | x)} [ \log p(z | x) - \log q(z; \lambda) ]

    based on importance sampling.

    Computed as

    .. math::
      1/B \sum_{b=1}^B [ w_{norm}(z^b; \lambda) *
                (\log p(x, z^b) - \log q(z^b; \lambda) ]

    where

    .. math::
      z^b \sim q(z^b; \lambda)

      w_{norm}(z^b; \lambda) = w(z^b; \lambda) / \sum_{b=1}^B (w(z^b; \lambda))

      w(z^b; \lambda) = p(x, z^b) / q(z^b; \lambda)

    which gives a gradient

    .. math::
      - 1/B \sum_{b=1}^B
      w_{norm}(z^b; \lambda) \partial_{\lambda} \log q(z^b; \lambda)

    """
    p_log_prob = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    for s in range(self.n_samples):
      z_sample = {}
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope='inference_' + str(s))
        z_sample[z] = qz_copy.value()
        q_log_prob[s] += tf.reduce_sum(
            qz.log_prob(tf.stop_gradient(z_sample[z])))

      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on posterior sample or
      # observed data.
      dict_swap = z_sample
      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          dict_swap[x] = obs

      if self.model_wrapper is None:
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope='inference_' + str(s))
          p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

        for x, obs in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope='inference_' + str(s))
            p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(obs))
      else:
        x = self.data
        p_log_prob[s] = self.model_wrapper.log_prob(x, z_sample)

    p_log_prob = tf.pack(p_log_prob)
    q_log_prob = tf.pack(q_log_prob)

    log_w = p_log_prob - q_log_prob
    log_w_norm = log_w - log_sum_exp(log_w)
    w_norm = tf.exp(log_w_norm)

    self.loss = tf.reduce_mean(w_norm * log_w)
    return -tf.reduce_mean(q_log_prob * tf.stop_gradient(w_norm))
