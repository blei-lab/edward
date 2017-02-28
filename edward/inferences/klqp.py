from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
import warnings

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable, Normal
from edward.util import copy, kl_multivariate_normal


class KLqp(VariationalInference):
  """Variational inference with the KL divergence

  .. math::

    \\text{KL}( q(z; \lambda) \| p(z \mid x) ).

  This class minimizes the objective by automatically selecting from a
  variety of black box inference techniques.

  Notes
  -----
  ``KLqp`` also optimizes any model parameters :math:`p(z \mid x;
  \\theta)`. It does this by variational EM, minimizing

  .. math::

    \mathbb{E}_{q(z; \lambda)} [ \log p(x, z; \\theta) ]

  with respect to :math:`\\theta`.

  In conditional inference, we infer :math:`z` in :math:`p(z, \\beta
  \mid x)` while fixing inference over :math:`\\beta` using another
  distribution :math:`q(\\beta)`. During gradient calculation, instead
  of using the model's density

  .. math::

    \log p(x, z^{(s)}), z^{(s)} \sim q(z; \lambda),

  for each sample :math:`s=1,\ldots,S`, ``KLqp`` uses

  .. math::

    \log p(x, z^{(s)}, \\beta^{(s)}),

  where :math:`z^{(s)} \sim q(z; \lambda)` and :math:`\\beta^{(s)}
  \sim q(\\beta)`.
  """
  def __init__(self, *args, **kwargs):
    super(KLqp, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, kl_scaling=None, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    kl_scaling: dict of RandomVariable to float, optional
      Provides option to scale terms when using ELBO with KL divergence.
      If the KL divergence terms are:
      .. math::
        \alpha_p E_{q(z| x, \lambda)} [ \log q(z| x, \lambda)/p(z)] then

      pass {p(z): \alpha_p} as kl_scaling, where \alpha_p is a float
      that specifies how much to scale the KL term.
    """
    if kl_scaling is None:
      kl_scaling = {}

    self.n_samples = n_samples
    self.kl_scaling = kl_scaling
    return super(KLqp, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    """Wrapper for the ``KLqp`` loss function.

    .. math::

      -\\text{ELBO} =
        -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]

    KLqp supports

    1. score function gradients (Paisley et al., 2012)
    2. reparameterization gradients (Kingma and Welling, 2014)

    of the loss function.

    If the variational model is a normal distribution and the prior is
    standard normal, then loss function can be written as

    .. math::

      -\mathbb{E}_{q(z; \lambda)}[\log p(x \mid z)] +
        \\text{KL}( q(z; \lambda) \| p(z) ),

    where the KL term is computed analytically (Kingma and Welling,
    2014).
    """
    is_reparameterizable = all([rv.is_reparameterized and rv.is_continuous
                                for rv in six.itervalues(self.latent_vars)])
    qz_is_normal = all([isinstance(rv, Normal) for
                       rv in six.itervalues(self.latent_vars)])
    z_is_normal = all([isinstance(rv, Normal) for
                       rv in six.iterkeys(self.latent_vars)])
    is_analytic_kl = qz_is_normal and \
        (z_is_normal or hasattr(self.model_wrapper, 'log_lik'))
    if is_reparameterizable:
      if is_analytic_kl:
        return build_reparam_kl_loss_and_gradients(self, var_list)
      # elif is_analytic_entropy:
      #    return build_reparam_entropy_loss_and_gradients(self, var_list)
      else:
        return build_reparam_loss_and_gradients(self, var_list)
    else:
      if is_analytic_kl:
        return build_score_kl_loss_and_gradients(self, var_list)
      # Analytic entropies may lead to problems around
      # convergence; for now it is deactivated.
      # elif is_analytic_entropy:
      #    return build_score_entropy_loss_and_gradients(self, var_list)
      else:
        return build_score_loss_and_gradients(self, var_list)


def MFVI(*args, **kwargs):
  warnings.simplefilter('default', DeprecationWarning)
  warnings.warn("MFVI is deprecated; use KLqp instead.", DeprecationWarning)
  return KLqp(*args, **kwargs)


class ReparameterizationKLqp(VariationalInference):
  """Variational inference with the KL divergence

  .. math::

    \\text{KL}( q(z; \lambda) \| p(z \mid x) ).

  This class minimizes the objective using the reparameterization
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

  def build_loss_and_gradients(self, var_list):
    return build_reparam_loss_and_gradients(self, var_list)


class ReparameterizationKLKLqp(VariationalInference):
  """Variational inference with the KL divergence

  .. math::

    \\text{KL}( q(z; \lambda) \| p(z \mid x) ).

  This class minimizes the objective using the reparameterization
  gradient and an analytic KL term.
  """
  def __init__(self, *args, **kwargs):
    super(ReparameterizationKLKLqp, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, kl_scaling=None, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    kl_scaling: dict of RandomVariable to float, optional
      Provides option to scale terms when using ELBO with KL divergence.
      If the KL divergence terms are:
      .. math::
        \alpha_p E_{q(z| x, \lambda)} [ \log q(z| x, \lambda)/p(z)] then

      pass {p(z): \alpha_p} as kl_scaling, where \alpha_p is a float
      that specifies how much to scale the KL term.
    """
    if kl_scaling is None:
      kl_scaling = {}

    self.n_samples = n_samples
    self.kl_scaling = kl_scaling
    return super(ReparameterizationKLKLqp, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    return build_reparam_kl_loss_and_gradients(self, var_list)


class ReparameterizationEntropyKLqp(VariationalInference):
  """Variational inference with the KL divergence

  .. math::

    \\text{KL}( q(z; \lambda) \| p(z \mid x) ).

  This class minimizes the objective using the reparameterization
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

  def build_loss_and_gradients(self, var_list):
    return build_reparam_entropy_loss_and_gradients(self, var_list)


class ScoreKLqp(VariationalInference):
  """Variational inference with the KL divergence

  .. math::

    \\text{KL}( q(z; \lambda) \| p(z \mid x) ).

  This class minimizes the objective using the score function
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

  def build_loss_and_gradients(self, var_list):
    return build_score_loss_and_gradients(self, var_list)


class ScoreKLKLqp(VariationalInference):
  """Variational inference with the KL divergence

  .. math::

    \\text{KL}( q(z; \lambda) \| p(z \mid x) ).

  This class minimizes the objective using the score function gradient
  and an analytic KL term.
  """
  def __init__(self, *args, **kwargs):
    super(ScoreKLKLqp, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, kl_scaling=None, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    kl_scaling: dict of RandomVariable to float, optional
      Provides option to scale terms when using ELBO with KL divergence.
      If the KL divergence terms are:
      .. math::
        \alpha_p E_{q(z| x, \lambda)} [ \log q(z| x, \lambda)/p(z)] then

      pass {p(z): \alpha_p} as kl_scaling, where \alpha_p is a float
      that specifies how much to scale the KL term.
    """
    if kl_scaling is None:
      kl_scaling = {}

    self.n_samples = n_samples
    self.kl_scaling = kl_scaling
    return super(ScoreKLKLqp, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    return build_score_kl_loss_and_gradients(self, var_list)


class ScoreEntropyKLqp(VariationalInference):
  """Variational inference with the KL divergence

  .. math::

    \\text{KL}( q(z; \lambda) \| p(z \mid x) ).

  This class minimizes the objective using the score function gradient
  and an analytic entropy term.
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

  def build_loss_and_gradients(self, var_list):
    return build_score_entropy_loss_and_gradients(self, var_list)


def build_reparam_loss_and_gradients(inference, var_list):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]

  based on the reparameterization trick (Kingma and Welling, 2014).

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * inference.n_samples
  q_log_prob = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    scope = 'inference_' + str(id(inference)) + '/' + str(s)
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      z_sample[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(
          inference.scale.get(z, 1.0) * qz_copy.log_prob(z_sample[z]))

    if inference.model_wrapper is None:
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      dict_swap = z_sample
      for x, qx in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      for z in six.iterkeys(inference.latent_vars):
        z_copy = copy(z, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            inference.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

      for x in six.iterkeys(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          p_log_prob[s] += tf.reduce_sum(
              inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))
    else:
      x = inference.data
      p_log_prob[s] = inference.model_wrapper.log_prob(x, z_sample)

  p_log_prob = tf.stack(p_log_prob)
  q_log_prob = tf.stack(q_log_prob)
  loss = -tf.reduce_mean(p_log_prob - q_log_prob)

  if var_list is None:
    var_list = tf.trainable_variables()

  grads = tf.gradients(loss, [v._ref() for v in var_list])
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars


def build_reparam_kl_loss_and_gradients(inference, var_list):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -\\text{ELBO} =  - ( \mathbb{E}_{q(z; \lambda)} [ \log p(x \mid z) ]
          + \\text{KL}(q(z; \lambda) \| p(z)) )

  based on the reparameterization trick (Kingma and Welling, 2014).

  It assumes the KL is analytic.

  For model wrappers, it assumes the prior is :math:`p(z) =
  \mathcal{N}(z; 0, 1)`.

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_lik = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    scope = 'inference_' + str(id(inference)) + '/' + str(s)
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      z_sample[z] = qz_copy.value()

    if inference.model_wrapper is None:
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      dict_swap = z_sample
      for x, qx in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      for x in six.iterkeys(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          p_log_lik[s] += tf.reduce_sum(
              inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))
    else:
      x = inference.data
      p_log_lik[s] = inference.model_wrapper.log_lik(x, z_sample)

  p_log_lik = tf.stack(p_log_lik)

  if inference.model_wrapper is None:
    kl = tf.reduce_sum([
        inference.kl_scaling.get(z, 1.0) * tf.reduce_sum(
            kl_multivariate_normal(qz.mu, qz.sigma, z.mu, z.sigma))
        for z, qz in six.iteritems(inference.latent_vars)])
  else:
    kl = tf.reduce_sum([tf.reduce_sum(kl_multivariate_normal(qz.mu, qz.sigma))
                        for qz in six.itervalues(inference.latent_vars)])

  loss = -(tf.reduce_mean(p_log_lik) - kl)

  if var_list is None:
    var_list = tf.trainable_variables()

  grads = tf.gradients(loss, [v._ref() for v in var_list])
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars


def build_reparam_entropy_loss_and_gradients(inference, var_list):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -\\text{ELBO} =  -( \mathbb{E}_{q(z; \lambda)} [ \log p(x , z) ]
          + \mathbb{H}(q(z; \lambda)) )

  based on the reparameterization trick (Kingma and Welling, 2014).

  It assumes the entropy is analytic.

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    scope = 'inference_' + str(id(inference)) + '/' + str(s)
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      z_sample[z] = qz_copy.value()

    if inference.model_wrapper is None:
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      dict_swap = z_sample
      for x, qx in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      for z in six.iterkeys(inference.latent_vars):
        z_copy = copy(z, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            inference.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

      for x in six.iterkeys(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          p_log_prob[s] += tf.reduce_sum(
              inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))
    else:
      x = inference.data
      p_log_prob[s] = inference.model_wrapper.log_prob(x, z_sample)

  p_log_prob = tf.stack(p_log_prob)

  q_entropy = tf.reduce_sum([inference.data.get(z, 1.0) * qz.entropy()
                             for z, qz in six.iteritems(inference.latent_vars)])

  loss = -(tf.reduce_mean(p_log_prob) + q_entropy)

  if var_list is None:
    var_list = tf.trainable_variables()

  grads = tf.gradients(loss, [v._ref() for v in var_list])
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars


def build_score_loss_and_gradients(inference, var_list):
  """Build loss function and gradients based on the score function
  estimator (Paisley et al., 2012).

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * inference.n_samples
  q_log_prob = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    scope = 'inference_' + str(id(inference)) + '/' + str(s)
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      z_sample[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(
          inference.scale.get(z, 1.0) *
          qz_copy.log_prob(tf.stop_gradient(z_sample[z])))

    if inference.model_wrapper is None:
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      dict_swap = z_sample
      for x, qx in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      for z in six.iterkeys(inference.latent_vars):
        z_copy = copy(z, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            inference.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

      for x in six.iterkeys(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          p_log_prob[s] += tf.reduce_sum(
              inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))
    else:
      x = inference.data
      p_log_prob[s] = inference.model_wrapper.log_prob(x, z_sample)

  p_log_prob = tf.stack(p_log_prob)
  q_log_prob = tf.stack(q_log_prob)

  if var_list is None:
    var_list = tf.trainable_variables()

  losses = p_log_prob - q_log_prob
  loss = -tf.reduce_mean(losses)

  grads = tf.gradients(
      -tf.reduce_mean(q_log_prob * tf.stop_gradient(losses)),
      [v._ref() for v in var_list])
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars


def build_score_kl_loss_and_gradients(inference, var_list):
  """Build loss function and gradients based on the score function
  estimator (Paisley et al., 2012).

  It assumes the KL is analytic.

  For model wrappers, it assumes the prior is :math:`p(z) =
  \mathcal{N}(z; 0, 1)`.

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_lik = [0.0] * inference.n_samples
  q_log_prob = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    scope = 'inference_' + str(id(inference)) + '/' + str(s)
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      z_sample[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(
          inference.scale.get(z, 1.0) *
          qz_copy.log_prob(tf.stop_gradient(z_sample[z])))

    if inference.model_wrapper is None:
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      dict_swap = z_sample
      for x, qx in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      for x in six.iterkeys(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          p_log_lik[s] += tf.reduce_sum(
              inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))
    else:
      x = inference.data
      p_log_lik[s] = inference.model_wrapper.log_lik(x, z_sample)

  p_log_lik = tf.stack(p_log_lik)
  q_log_prob = tf.stack(q_log_prob)

  if inference.model_wrapper is None:
    kl = tf.reduce_sum([
        inference.kl_scaling.get(z, 1.0) * tf.reduce_sum(
            kl_multivariate_normal(qz.mu, qz.sigma, z.mu, z.sigma))
        for z, qz in six.iteritems(inference.latent_vars)])
  else:
    kl = tf.reduce_sum([tf.reduce_sum(kl_multivariate_normal(qz.mu, qz.sigma))
                        for qz in six.itervalues(inference.latent_vars)])

  if var_list is None:
    var_list = tf.trainable_variables()

  loss = -(tf.reduce_mean(p_log_lik) - kl)
  grads = tf.gradients(
      -(tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_lik)) - kl),
      [v._ref() for v in var_list])
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars


def build_score_entropy_loss_and_gradients(inference, var_list):
  """Build loss function and gradients based on the score function
  estimator (Paisley et al., 2012).

  It assumes the entropy is analytic.

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * inference.n_samples
  q_log_prob = [0.0] * inference.n_samples
  for s in range(inference.n_samples):
    scope = 'inference_' + str(id(inference)) + '/' + str(s)
    z_sample = {}
    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      z_sample[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(
          inference.scale.get(z, 1.0) *
          qz_copy.log_prob(tf.stop_gradient(z_sample[z])))

    if inference.model_wrapper is None:
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      dict_swap = z_sample
      for x, qx in six.iteritems(inference.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      for z in six.iterkeys(inference.latent_vars):
        z_copy = copy(z, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            inference.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

      for x in six.iterkeys(inference.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          p_log_prob[s] += tf.reduce_sum(
              inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))
    else:
      x = inference.data
      p_log_prob[s] = inference.model_wrapper.log_prob(x, z_sample)

  p_log_prob = tf.stack(p_log_prob)
  q_log_prob = tf.stack(q_log_prob)

  q_entropy = tf.reduce_sum([inference.data.get(z, 1.0) * qz.entropy()
                             for z, qz in six.iteritems(inference.latent_vars)])

  if var_list is None:
    var_list = tf.trainable_variables()

  loss = -(tf.reduce_mean(p_log_prob) + q_entropy)
  grads = tf.gradients(
      -(tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_prob)) +
          q_entropy),
      [v._ref() for v in var_list])
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars
