from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable, Normal
from edward.util import copy, kl_multivariate_normal


class KLqp(VariationalInference):
  """Variational inference with the KL divergence.

  This class implements a variety of "black-box" variational inference
  techniques (Ranganath et al., 2014) that minimize

  .. math::

    KL( q(z; \lambda) || p(z | x) ).

  This is equivalent to maximizing the objective function (Jordan et al., 1999)

  .. math::

    ELBO =  E_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ].
  """
  def __init__(self, *args, **kwargs):
    super(KLqp, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, score=None, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    score : bool, optional
      Whether to force inference to use the score function
      gradient estimator. Otherwise default is to use the
      reparameterization gradient if available.
    """
    if score is None and \
       all([rv.is_reparameterized and rv.is_continuous
            for rv in six.itervalues(self.latent_vars)]):
      self.score = False
    else:
      self.score = True

    self.n_samples = n_samples
    return super(KLqp, self).initialize(*args, **kwargs)

  def build_loss(self):
    """Wrapper for the KLqp loss function.

    .. math::

      -ELBO =  -E_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]

    KLqp supports

    1. score function gradients
    2. reparameterization gradients

    of the loss function.

    If the variational model is a Gaussian distribution, then part of the
    loss function can be computed analytically.

    If the variational model is a normal distribution and the prior is
    standard normal, then part of the loss function can be computed
    analytically following Kingma and Welling (2014),

    .. math::

      E[\log p(x | z) + KL],

    where the KL term is computed analytically.

    Returns
    -------
    result :
      an appropriately selected loss function form
    """
    qz_is_normal = all([isinstance(rv, Normal) for
                       rv in six.itervalues(self.latent_vars)])
    z_is_normal = all([isinstance(rv, Normal) for
                       rv in six.iterkeys(self.latent_vars)])
    is_analytic_kl = qz_is_normal and \
        (z_is_normal or hasattr(self.model_wrapper, 'log_lik'))
    if self.score:
      if is_analytic_kl:
        return build_score_loss_kl(self)
      # Analytic entropies may lead to problems around
      # convergence; for now it is deactivated.
      # elif is_analytic_entropy:
      #    return build_score_loss_entropy(self)
      else:
        return build_score_loss(self)
    else:
      if is_analytic_kl:
        return build_reparam_loss_kl(self)
      # elif is_analytic_entropy:
      #    return build_reparam_loss_entropy(self)
      else:
        return build_reparam_loss(self)


MFVI = KLqp  # deprecated synonym


class ReparameterizationKLqp(VariationalInference):
  """Variational inference with the KL divergence.

  This class minimizes the KL divergence using the reparameterization
  gradient.
  """
  def __init__(self, *args, **kwargs):
    super(ReparameterizationKLqp, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    """
    self.n_samples = n_samples
    return super(ReparameterizationKLqp, self).initialize(*args, **kwargs)

  def build_loss(self):
    return build_reparam_loss(inference)


class ReparameterizationKLKLqp(VariationalInference):
  """Variational inference with the KL divergence.

  This class minimizes the KL divergence using the reparameterization
  gradient and an analytic KL term.
  """
  def __init__(self, *args, **kwargs):
    super(ReparameterizationKLKLqp, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    """
    self.n_samples = n_samples
    return super(ReparameterizationKLKLqp, self).initialize(*args, **kwargs)

  def build_loss(self):
    return build_reparam_kl_loss(inference)


class ReparameterizationEntropyKLqp(VariationalInference):
  """Variational inference with the KL divergence.

  This class minimizes the KL divergence using the reparameterization
  gradient and an analytic entropy term.
  """
  def __init__(self, *args, **kwargs):
    super(ReparameterizationEntropyKLqp, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    """
    self.n_samples = n_samples
    return super(ReparameterizationEntropyKLqp, self).initialize(
        *args, **kwargs)

  def build_loss(self):
    return build_reparam_entropy_loss(inference)


class ScoreKLqp(VariationalInference):
  """Variational inference with the KL divergence.

  This class minimizes the KL divergence using the score function
  gradient.
  """
  def __init__(self, *args, **kwargs):
    super(ScoreKLqp, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    """
    self.n_samples = n_samples
    return super(ScoreKLqp, self).initialize(*args, **kwargs)

  def build_loss(self):
    return build_score_loss(inference)


class ScoreKLKLqp(VariationalInference):
  """Variational inference with the KL divergence.

  This class minimizes the KL divergence using the score function
  gradient and an analytic KL term.
  """
  def __init__(self, *args, **kwargs):
    super(ScoreKLKLqp, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    """
    self.n_samples = n_samples
    return super(ScoreKLKLqp, self).initialize(*args, **kwargs)

  def build_loss(self):
    return build_score_kl_loss(inference)


class ScoreEntropyKLqp(VariationalInference):
  """Variational inference with the KL divergence.

  This class minimizes the KL divergence using the score function
  gradient and an analytic entropy term.
  """
  def __init__(self, *args, **kwargs):
    super(ScoreEntropyKLqp, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    """
    self.n_samples = n_samples
    return super(ScoreEntropyKLqp, self).initialize(*args, **kwargs)

  def build_loss(self):
    return build_score_entropy_loss(inference)


def build_reparam_loss(inference):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -ELBO =  -E_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]

  based on the reparameterization trick. (Kingma and Welling, 2014)

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * inference.n_samples
  q_log_prob = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope='inference_' + str(s))
      z_sample[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(qz.log_prob(z_sample[z]))

    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on posterior sample or
    # observed data.
    dict_swap = z_sample
    for x, obs in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        dict_swap[x] = obs

    if inference.model_wrapper is None:
      for z in six.iterkeys(inference.latent_vars):
        z_copy = copy(z, dict_swap, scope='inference_' + str(s))
        p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

      for x, obs in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope='inference_' + str(s))
          p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(obs))
    else:
      x = inference.data
      p_log_prob[s] = inference.model_wrapper.log_prob(x, z_sample)

  p_log_prob = tf.pack(p_log_prob)
  q_log_prob = tf.pack(q_log_prob)
  inference.loss = tf.reduce_mean(p_log_prob - q_log_prob)
  return -inference.loss


def build_reparam_loss_kl(inference):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -ELBO =  - ( E_{q(z; \lambda)} [ \log p(x | z) ]
          + KL(q(z; \lambda) || p(z)) )

  based on the reparameterization trick. (Kingma and Welling, 2014)

  It assumes the KL is analytic.

  For model wrappers, it assumes the prior is :math:`p(z) =
  \mathcal{N}(z; 0, 1)`.

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_lik = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope='inference_' + str(s))
      z_sample[z] = qz_copy.value()

    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on posterior sample or
    # observed data.
    dict_swap = z_sample
    for x, obs in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        dict_swap[x] = obs

    if inference.model_wrapper is None:
      for x, obs in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope='inference_' + str(s))
          p_log_lik[s] += tf.reduce_sum(x_copy.log_prob(obs))
    else:
      x = inference.data
      p_log_lik[s] = inference.model_wrapper.log_lik(x, z_sample)

  p_log_lik = tf.pack(p_log_lik)

  if inference.model_wrapper is None:
    kl = tf.reduce_sum([tf.reduce_sum(kl_multivariate_normal(
                        qz.mu, qz.sigma, z.mu, z.sigma))
                        for z, qz in six.iteritems(inference.latent_vars)])
  else:
    kl = tf.reduce_sum([tf.reduce_sum(kl_multivariate_normal(qz.mu, qz.sigma))
                        for qz in six.itervalues(inference.latent_vars)])

  p_log_lik = tf.pack(p_log_lik)
  inference.loss = tf.reduce_mean(p_log_lik) - kl
  return -inference.loss


def build_reparam_loss_entropy(inference):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -ELBO =  -( E_{q(z; \lambda)} [ \log p(x , z) ]
          + H(q(z; \lambda)) )

  based on the reparameterization trick. (Kingma and Welling, 2014)

  It assumes the entropy is analytic.

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope='inference_' + str(s))
      z_sample[z] = qz_copy.value()

    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on posterior sample or
    # observed data.
    dict_swap = z_sample
    for x, obs in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        dict_swap[x] = obs

    if inference.model_wrapper is None:
      for z in six.iterkeys(inference.latent_vars):
        z_copy = copy(z, dict_swap, scope='inference_' + str(s))
        p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

      for x, obs in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope='inference_' + str(s))
          p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(obs))
    else:
      x = inference.data
      p_log_prob[s] = inference.model_wrapper.log_prob(x, z_sample)

  p_log_prob = tf.pack(p_log_prob)

  q_entropy = tf.reduce_sum([qz.entropy()
                             for qz in six.itervalues(inference.latent_vars)])

  inference.loss = tf.reduce_mean(p_log_prob) + q_entropy
  return -inference.loss


def build_score_loss(inference):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -ELBO =  -E_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]

  based on the score function estimator. (Paisley et al., 2012)

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * inference.n_samples
  q_log_prob = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope='inference_' + str(s))
      z_sample[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(
          qz.log_prob(tf.stop_gradient(z_sample[z])))

    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on posterior sample or
    # observed data.
    dict_swap = z_sample
    for x, obs in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        dict_swap[x] = obs

    if inference.model_wrapper is None:
      for z in six.iterkeys(inference.latent_vars):
        z_copy = copy(z, dict_swap, scope='inference_' + str(s))
        p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

      for x, obs in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope='inference_' + str(s))
          p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(obs))
    else:
      x = inference.data
      p_log_prob[s] = inference.model_wrapper.log_prob(x, z_sample)

  p_log_prob = tf.pack(p_log_prob)
  q_log_prob = tf.pack(q_log_prob)

  losses = p_log_prob - q_log_prob
  inference.loss = tf.reduce_mean(losses)
  return -tf.reduce_mean(q_log_prob * tf.stop_gradient(losses))


def build_score_loss_kl(inference):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -ELBO =  - ( E_{q(z; \lambda)} [ \log p(x | z) ]
           + KL(q(z; \lambda) || p(z)) )

  based on the score function estimator. (Paisley et al., 2012)

  It assumes the KL is analytic.

  For model wrappers, it assumes the prior is :math:`p(z) =
  \mathcal{N}(z; 0, 1)`.

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_lik = [0.0] * inference.n_samples
  q_log_prob = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope='inference_' + str(s))
      z_sample[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(
          qz.log_prob(tf.stop_gradient(z_sample[z])))

    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on posterior sample or
    # observed data.
    dict_swap = z_sample
    for x, obs in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        dict_swap[x] = obs

    if inference.model_wrapper is None:
      for x, obs in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope='inference_' + str(s))
          p_log_lik[s] += tf.reduce_sum(x_copy.log_prob(obs))
    else:
      x = inference.data
      p_log_lik[s] = inference.model_wrapper.log_lik(x, z_sample)

  p_log_lik = tf.pack(p_log_lik)
  q_log_prob = tf.pack(q_log_prob)

  if inference.model_wrapper is None:
    kl = tf.reduce_sum([tf.reduce_sum(kl_multivariate_normal(
                        qz.mu, qz.sigma, z.mu, z.sigma))
                        for z, qz in six.iteritems(inference.latent_vars)])
  else:
    kl = tf.reduce_sum([tf.reduce_sum(kl_multivariate_normal(qz.mu, qz.sigma))
                        for qz in six.itervalues(inference.latent_vars)])

  inference.loss = tf.reduce_mean(p_log_lik) - kl
  return -(tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_lik)) - kl)


def build_score_loss_entropy(inference):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -ELBO =  - ( E_{q(z; \lambda)} [ \log p(x, z) ]
          + H(q(z; \lambda)) )

  based on the score function estimator. (Paisley et al., 2012)

  It assumes the entropy is analytic.

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * inference.n_samples
  q_log_prob = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope='inference_' + str(s))
      z_sample[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(
          qz.log_prob(tf.stop_gradient(z_sample[z])))

    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on posterior sample or
    # observed data.
    dict_swap = z_sample
    for x, obs in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        dict_swap[x] = obs

    if inference.model_wrapper is None:
      for z in six.iterkeys(inference.latent_vars):
        z_copy = copy(z, dict_swap, scope='inference_' + str(s))
        p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

      for x, obs in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope='inference_' + str(s))
          p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(obs))
    else:
      x = inference.data
      p_log_prob[s] = inference.model_wrapper.log_prob(x, z_sample)

  p_log_prob = tf.pack(p_log_prob)
  q_log_prob = tf.pack(q_log_prob)

  q_entropy = tf.reduce_sum([qz.entropy()
                             for qz in six.itervalues(inference.latent_vars)])

  inference.loss = tf.reduce_mean(p_log_prob) + q_entropy
  return -(tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_prob)) +
           q_entropy)
