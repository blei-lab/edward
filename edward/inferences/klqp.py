from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import (check_and_maybe_build_data,
    check_and_maybe_build_latent_vars, transform, check_and_maybe_build_dict, check_and_maybe_build_var_list)
from edward.models import RandomVariable
from edward.util import copy, get_descendants

try:
  from edward.models import Normal
  from tensorflow.contrib.distributions import kl_divergence
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


def klqp(latent_vars=None, data=None, n_samples=1, kl_scaling=None,
         auto_transform=True, scale=None, var_list=None, summary_key=None):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective by automatically selecting from a
  variety of black box inference techniques.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized using a
      `Normal` random variable that is defined internally with a
      free parameter per location and scale and is initialized using
      standard normal draws. The random variables to approximate
      must be continuous.
    n_samples: int, optional.
      Number of samples from variational model for calculating
      stochastic gradients.
    kl_scaling: dict of RandomVariable to tf.Tensor, optional.
      Provides option to scale terms when using ELBO with KL divergence.
      If the KL divergence terms are

      $\\alpha_p \mathbb{E}_{q(z\mid x, \lambda)} [
            \log q(z\mid x, \lambda) - \log p(z)],$

      then pass {$p(z)$: $\\alpha_p$} as `kl_scaling`,
      where $\\alpha_p$ is a tensor. Its shape must be broadcastable;
      it is multiplied element-wise to the batchwise KL terms.
    scale: dict of RandomVariable to tf.Tensor, optional.
      A tensor to dict computation for any random variable that it is
      binded to. Its shape must be broadcastable; it is multiplied
      element-wise to the random variable. For example, this is useful
      for mini-batch scaling when inferring global variables, or
      applying masks on a random variable.

  #### Notes

  `KLqp` also optimizes any model parameters $p(z \mid x;
  \\theta)$. It does this by variational EM, maximizing

  $\mathbb{E}_{q(z; \lambda)} [ \log p(x, z; \\theta) ]$

  with respect to $\\theta$.

  In conditional inference, we infer $z$ in $p(z, \\beta
  \mid x)$ while fixing inference over $\\beta$ using another
  distribution $q(\\beta)$. During gradient calculation, instead
  of using the model's density

  $\log p(x, z^{(s)}), z^{(s)} \sim q(z; \lambda),$

  for each sample $s=1,\ldots,S$, `KLqp` uses

  $\log p(x, z^{(s)}, \\beta^{(s)}),$

  where $z^{(s)} \sim q(z; \lambda)$ and $\\beta^{(s)}
  \sim q(\\beta)$.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  ##

  $-\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

  KLqp supports

  1. score function gradients [@paisley2012variational]
  2. reparameterization gradients [@kingma2014auto]

  of the loss function.

  If the KL divergence between the variational model and the prior
  is tractable, then the loss function can be written as

  $-\mathbb{E}_{q(z; \lambda)}[\log p(x \mid z)] +
      \\text{KL}( q(z; \lambda) \| p(z) ),$

  where the KL term is computed analytically [@kingma2014auto]. We
  compute this automatically when $p(z)$ and $q(z; \lambda)$ are
  Normal.
  """
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars_dict = {}
      continuous = \
          ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
      for z in latent_vars:
        if not hasattr(z, 'support') or z.support not in continuous:
          raise AttributeError(
              "Random variable {} is not continuous or a random "
              "variable with supported continuous support.".format(z))
        batch_event_shape = z.batch_shape.concatenate(z.event_shape)
        loc = tf.Variable(tf.random_normal(batch_event_shape))
        scale = tf.nn.softplus(
            tf.Variable(tf.random_normal(batch_event_shape)))
        latent_vars_dict[z] = Normal(loc=loc, scale=scale)
      latent_vars = latent_vars_dict
      del latent_vars_dict
  latent_vars = check_and_maybe_build_latent_vars(latent_vars)
  data = check_and_maybe_build_data(data)
  latent_vars, _ = transform(latent_vars, auto_transform)
  kl_scaling = check_and_maybe_build_dict(kl_scaling)
  scale = check_and_maybe_build_dict(scale)
  var_list = check_and_maybe_build_var_list(var_list, latent_vars, data)

  is_reparameterizable = all([
      rv.reparameterization_type ==
      tf.contrib.distributions.FULLY_REPARAMETERIZED
      for rv in six.itervalues(latent_vars)])
  is_analytic_kl = all([isinstance(z, Normal) and isinstance(qz, Normal)
                        for z, qz in six.iteritems(latent_vars)])
  if not is_analytic_kl and kl_scaling:
    raise TypeError("kl_scaling must be None when using non-analytic KL term")
  if is_reparameterizable:
    if is_analytic_kl:
      return build_reparam_kl_loss_and_gradients(
          latent_vars, data, var_list,
          scale, n_samples, kl_scaling, summary_key)
    # elif is_analytic_entropy:
    #    return build_reparam_entropy_loss_and_gradients(...)
    else:
      return build_reparam_loss_and_gradients(
          latent_vars, data, var_list,
          scale, n_samples, summary_key)
  else:
    # Prefer Rao-Blackwellization over analytic KL. Unknown what
    # would happen stability-wise if the two are combined.
    # if is_analytic_kl:
    #   return build_score_kl_loss_and_gradients(...)
    # Analytic entropies may lead to problems around
    # convergence; for now it is deactivated.
    # elif is_analytic_entropy:
    #    return build_score_entropy_loss_and_gradients(...)
    # else:
    return build_score_rb_loss_and_gradients(
        latent_vars, data, var_list,
        scale, n_samples, summary_key)


def reparameterization_klqp(
    latent_vars=None, data=None, n_samples=1,
    auto_transform=True, scale=None, var_list=None, summary_key=None):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the reparameterization
  gradient.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized using a
      `Normal` random variable that is defined internally with a
      free parameter per location and scale and is initialized using
      standard normal draws. The random variables to approximate
      must be continuous.
    n_samples: int, optional.
      Number of samples from variational model for calculating
      stochastic gradients.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars_dict = {}
      continuous = \
          ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
      for z in latent_vars:
        if not hasattr(z, 'support') or z.support not in continuous:
          raise AttributeError(
              "Random variable {} is not continuous or a random "
              "variable with supported continuous support.".format(z))
        batch_event_shape = z.batch_shape.concatenate(z.event_shape)
        loc = tf.Variable(tf.random_normal(batch_event_shape))
        scale = tf.nn.softplus(
            tf.Variable(tf.random_normal(batch_event_shape)))
        latent_vars_dict[z] = Normal(loc=loc, scale=scale)
      latent_vars = latent_vars_dict
      del latent_vars_dict
  latent_vars = check_and_maybe_build_latent_vars(latent_vars)
  data = check_and_maybe_build_data(data)
  latent_vars, _ = transform(latent_vars, auto_transform)
  scale = check_and_maybe_build_dict(scale)
  var_list = check_and_maybe_build_var_list(var_list, latent_vars, data)

  return build_reparam_loss_and_gradients(
      latent_vars, data, var_list,
      scale, n_samples, summary_key)


def reparameterization_kl_klqp(
    latent_vars=None, data=None, n_samples=1, kl_scaling=None,
    auto_transform=True, scale=None, var_list=None, summary_key=None):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the reparameterization
  gradient and an analytic KL term.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized using a
      `Normal` random variable that is defined internally with a
      free parameter per location and scale and is initialized using
      standard normal draws. The random variables to approximate
      must be continuous.
    n_samples: int, optional.
      Number of samples from variational model for calculating
      stochastic gradients.
    kl_scaling: dict of RandomVariable to tf.Tensor, optional.
      Provides option to scale terms when using ELBO with KL divergence.
      If the KL divergence terms are

      $\\alpha_p \mathbb{E}_{q(z\mid x, \lambda)} [
            \log q(z\mid x, \lambda) - \log p(z)],$

      then pass {$p(z)$: $\\alpha_p$} as `kl_scaling`,
      where $\\alpha_p$ is a tensor. Its shape must be broadcastable;
      it is multiplied element-wise to the batchwise KL terms.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars_dict = {}
      continuous = \
          ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
      for z in latent_vars:
        if not hasattr(z, 'support') or z.support not in continuous:
          raise AttributeError(
              "Random variable {} is not continuous or a random "
              "variable with supported continuous support.".format(z))
        batch_event_shape = z.batch_shape.concatenate(z.event_shape)
        loc = tf.Variable(tf.random_normal(batch_event_shape))
        scale = tf.nn.softplus(
            tf.Variable(tf.random_normal(batch_event_shape)))
        latent_vars_dict[z] = Normal(loc=loc, scale=scale)
      latent_vars = latent_vars_dict
      del latent_vars_dict
  latent_vars = check_and_maybe_build_latent_vars(latent_vars)
  data = check_and_maybe_build_data(data)
  latent_vars, _ = transform(latent_vars, auto_transform)
  kl_scaling = check_and_maybe_build_dict(kl_scaling)
  scale = check_and_maybe_build_dict(scale)
  var_list = check_and_maybe_build_var_list(var_list, latent_vars, data)

  return build_reparam_kl_loss_and_gradients(
      latent_vars, data, var_list,
      scale, n_samples, kl_scaling, summary_key)


def reparameterization_entropy_klqp(
    latent_vars=None, data=None, n_samples=1,
    auto_transform=True, scale=None, var_list=None, summary_key=None):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the reparameterization
  gradient and an analytic entropy term.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized using a
      `Normal` random variable that is defined internally with a
      free parameter per location and scale and is initialized using
      standard normal draws. The random variables to approximate
      must be continuous.
    n_samples: int, optional.
      Number of samples from variational model for calculating
      stochastic gradients.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars_dict = {}
      continuous = \
          ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
      for z in latent_vars:
        if not hasattr(z, 'support') or z.support not in continuous:
          raise AttributeError(
              "Random variable {} is not continuous or a random "
              "variable with supported continuous support.".format(z))
        batch_event_shape = z.batch_shape.concatenate(z.event_shape)
        loc = tf.Variable(tf.random_normal(batch_event_shape))
        scale = tf.nn.softplus(
            tf.Variable(tf.random_normal(batch_event_shape)))
        latent_vars_dict[z] = Normal(loc=loc, scale=scale)
      latent_vars = latent_vars_dict
      del latent_vars_dict
  latent_vars = check_and_maybe_build_latent_vars(latent_vars)
  data = check_and_maybe_build_data(data)
  latent_vars, _ = transform(latent_vars, auto_transform)
  scale = check_and_maybe_build_dict(scale)
  var_list = check_and_maybe_build_var_list(var_list, latent_vars, data)

  return build_reparam_entropy_loss_and_gradients(
      latent_vars, data, var_list,
      scale, n_samples, summary_key)


def score_klqp(
    latent_vars=None, data=None, n_samples=1,
    auto_transform=True, scale=None, var_list=None, summary_key=None):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the score function
  gradient.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized using a
      `Normal` random variable that is defined internally with a
      free parameter per location and scale and is initialized using
      standard normal draws. The random variables to approximate
      must be continuous.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars_dict = {}
      continuous = \
          ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
      for z in latent_vars:
        if not hasattr(z, 'support') or z.support not in continuous:
          raise AttributeError(
              "Random variable {} is not continuous or a random "
              "variable with supported continuous support.".format(z))
        batch_event_shape = z.batch_shape.concatenate(z.event_shape)
        loc = tf.Variable(tf.random_normal(batch_event_shape))
        scale = tf.nn.softplus(
            tf.Variable(tf.random_normal(batch_event_shape)))
        latent_vars_dict[z] = Normal(loc=loc, scale=scale)
      latent_vars = latent_vars_dict
      del latent_vars_dict
  latent_vars = check_and_maybe_build_latent_vars(latent_vars)
  data = check_and_maybe_build_data(data)
  latent_vars, _ = transform(latent_vars, auto_transform)
  scale = check_and_maybe_build_dict(scale)
  var_list = check_and_maybe_build_var_list(var_list, latent_vars, data)

  return build_score_loss_and_gradients(
      latent_vars, data, var_list,
      scale, n_samples, summary_key)


def score_kl_klqp(
    latent_vars=None, data=None, n_samples=1, kl_scaling=None,
    auto_transform=True, scale=None, var_list=None, summary_key=None):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the score function gradient
  and an analytic KL term.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized using a
      `Normal` random variable that is defined internally with a
      free parameter per location and scale and is initialized using
      standard normal draws. The random variables to approximate
      must be continuous.
    n_samples: int, optional.
      Number of samples from variational model for calculating
      stochastic gradients.
    kl_scaling: dict of RandomVariable to tf.Tensor, optional.
      Provides option to scale terms when using ELBO with KL divergence.
      If the KL divergence terms are

      $\\alpha_p \mathbb{E}_{q(z\mid x, \lambda)} [
            \log q(z\mid x, \lambda) - \log p(z)],$

      then pass {$p(z)$: $\\alpha_p$} as `kl_scaling`,
      where $\\alpha_p$ is a tensor. Its shape must be broadcastable;
      it is multiplied element-wise to the batchwise KL terms.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars_dict = {}
      continuous = \
          ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
      for z in latent_vars:
        if not hasattr(z, 'support') or z.support not in continuous:
          raise AttributeError(
              "Random variable {} is not continuous or a random "
              "variable with supported continuous support.".format(z))
        batch_event_shape = z.batch_shape.concatenate(z.event_shape)
        loc = tf.Variable(tf.random_normal(batch_event_shape))
        scale = tf.nn.softplus(
            tf.Variable(tf.random_normal(batch_event_shape)))
        latent_vars_dict[z] = Normal(loc=loc, scale=scale)
      latent_vars = latent_vars_dict
      del latent_vars_dict
  latent_vars = check_and_maybe_build_latent_vars(latent_vars)
  data = check_and_maybe_build_data(data)
  latent_vars, _ = transform(latent_vars, auto_transform)
  kl_scaling = check_and_maybe_build_dict(kl_scaling)
  scale = check_and_maybe_build_dict(scale)
  var_list = check_and_maybe_build_var_list(var_list, latent_vars, data)

  return build_score_kl_loss_and_gradients(
      latent_vars, data, var_list,
      scale, n_samples, kl_scaling, summary_key)


def score_entropy_klqp(
    latent_vars=None, data=None, n_samples=1,
    auto_transform=True, scale=None, var_list=None, summary_key=None):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the score function gradient
  and an analytic entropy term.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized using a
      `Normal` random variable that is defined internally with a
      free parameter per location and scale and is initialized using
      standard normal draws. The random variables to approximate
      must be continuous.
    n_samples: int, optional.
      Number of samples from variational model for calculating
      stochastic gradients.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars_dict = {}
      continuous = \
          ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
      for z in latent_vars:
        if not hasattr(z, 'support') or z.support not in continuous:
          raise AttributeError(
              "Random variable {} is not continuous or a random "
              "variable with supported continuous support.".format(z))
        batch_event_shape = z.batch_shape.concatenate(z.event_shape)
        loc = tf.Variable(tf.random_normal(batch_event_shape))
        scale = tf.nn.softplus(
            tf.Variable(tf.random_normal(batch_event_shape)))
        latent_vars_dict[z] = Normal(loc=loc, scale=scale)
      latent_vars = latent_vars_dict
      del latent_vars_dict
  latent_vars = check_and_maybe_build_latent_vars(latent_vars)
  data = check_and_maybe_build_data(data)
  latent_vars, _ = transform(latent_vars, auto_transform)
  scale = check_and_maybe_build_dict(scale)
  var_list = check_and_maybe_build_var_list(var_list, latent_vars, data)

  return build_score_entropy_loss_and_gradients(
      latent_vars, data, var_list,
      scale, n_samples, summary_key)


def score_rb_klqp(
    latent_vars=None, data=None, n_samples=1,
    auto_transform=True, scale=None, var_list=None, summary_key=None):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the score function gradient
  and Rao-Blackwellization.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized using a
      `Normal` random variable that is defined internally with a
      free parameter per location and scale and is initialized using
      standard normal draws. The random variables to approximate
      must be continuous.

  #### Notes

  Current Rao-Blackwellization is limited to Rao-Blackwellizing across
  stochastic nodes in the computation graph. It does not
  Rao-Blackwellize within a node such as when a node represents
  multiple random variables via non-scalar batch shape.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars_dict = {}
      continuous = \
          ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
      for z in latent_vars:
        if not hasattr(z, 'support') or z.support not in continuous:
          raise AttributeError(
              "Random variable {} is not continuous or a random "
              "variable with supported continuous support.".format(z))
        batch_event_shape = z.batch_shape.concatenate(z.event_shape)
        loc = tf.Variable(tf.random_normal(batch_event_shape))
        scale = tf.nn.softplus(
            tf.Variable(tf.random_normal(batch_event_shape)))
        latent_vars_dict[z] = Normal(loc=loc, scale=scale)
      latent_vars = latent_vars_dict
      del latent_vars_dict
  latent_vars = check_and_maybe_build_latent_vars(latent_vars)
  data = check_and_maybe_build_data(data)
  latent_vars, _ = transform(latent_vars, auto_transform)
  scale = check_and_maybe_build_dict(scale)
  var_list = check_and_maybe_build_var_list(var_list, latent_vars, data)

  return build_score_rb_loss_and_gradients(
      latent_vars, data, var_list,
      scale, n_samples, summary_key)


def build_reparam_loss_and_gradients(
    latent_vars, data, var_list, scale, n_samples, summary_key):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  $-\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

  based on the reparameterization trick [@kingma2014auto].

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value
      q_log_prob[s] += tf.reduce_sum(
          scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))

    for z in six.iterkeys(latent_vars):
      z_copy = copy(z, dict_swap, scope=scope)
      p_log_prob[s] += tf.reduce_sum(
          scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

  p_log_prob = tf.reduce_mean(p_log_prob)
  q_log_prob = tf.reduce_mean(q_log_prob)
  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())

  if summary_key is not None:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=[summary_key])
    tf.summary.scalar("loss/q_log_prob", q_log_prob,
                      collections=[summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[summary_key])

  loss = -(p_log_prob - q_log_prob - reg_penalty)

  grads = tf.gradients(loss, var_list)
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars


def build_reparam_kl_loss_and_gradients(
    latent_vars, data, var_list, scale, n_samples, kl_scaling, summary_key):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -\\text{ELBO} =  - ( \mathbb{E}_{q(z; \lambda)} [ \log p(x \mid z) ]
          + \\text{KL}(q(z; \lambda) \| p(z)) )

  based on the reparameterization trick [@kingma2014auto].

  It assumes the KL is analytic.

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_lik = [0.0] * n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value

    for x in six.iterkeys(data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_lik[s] += tf.reduce_sum(
            scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

  p_log_lik = tf.reduce_mean(p_log_lik)

  kl_penalty = tf.reduce_sum([
      tf.reduce_sum(kl_scaling.get(z, 1.0) * kl_divergence(qz, z))
      for z, qz in six.iteritems(latent_vars)])

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())

  if summary_key is not None:
    tf.summary.scalar("loss/p_log_lik", p_log_lik,
                      collections=[summary_key])
    tf.summary.scalar("loss/kl_penalty", kl_penalty,
                      collections=[summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[summary_key])

  loss = -(p_log_lik - kl_penalty - reg_penalty)

  grads = tf.gradients(loss, var_list)
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars


def build_reparam_entropy_loss_and_gradients(
    latent_vars, data, var_list, scale, n_samples, summary_key):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  $-\\text{ELBO} =  -( \mathbb{E}_{q(z; \lambda)} [ \log p(x , z) ]
          + \mathbb{H}(q(z; \lambda)) )$

  based on the reparameterization trick [@kingma2014auto].

  It assumes the entropy is analytic.

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value

    for z in six.iterkeys(latent_vars):
      z_copy = copy(z, dict_swap, scope=scope)
      p_log_prob[s] += tf.reduce_sum(
          scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

  p_log_prob = tf.reduce_mean(p_log_prob)

  q_entropy = tf.reduce_sum([
      tf.reduce_sum(qz.entropy())
      for z, qz in six.iteritems(latent_vars)])

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())

  if summary_key is not None:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=[summary_key])
    tf.summary.scalar("loss/q_entropy", q_entropy,
                      collections=[summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[summary_key])

  loss = -(p_log_prob + q_entropy - reg_penalty)

  grads = tf.gradients(loss, var_list)
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars


def build_score_loss_and_gradients(
    latent_vars, data, var_list, scale, n_samples, summary_key):
  """Build loss function and gradients based on the score function
  estimator [@paisley2012variational].

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value
      q_log_prob[s] += tf.reduce_sum(
          scale.get(z, 1.0) *
          qz_copy.log_prob(tf.stop_gradient(dict_swap[z])))

    for z in six.iterkeys(latent_vars):
      z_copy = copy(z, dict_swap, scope=scope)
      p_log_prob[s] += tf.reduce_sum(
          scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

  p_log_prob = tf.stack(p_log_prob)
  q_log_prob = tf.stack(q_log_prob)
  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())

  if summary_key is not None:
    tf.summary.scalar("loss/p_log_prob", tf.reduce_mean(p_log_prob),
                      collections=[summary_key])
    tf.summary.scalar("loss/q_log_prob", tf.reduce_mean(q_log_prob),
                      collections=[summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[summary_key])

  losses = p_log_prob - q_log_prob
  loss = -(tf.reduce_mean(losses) - reg_penalty)

  q_rvs = list(six.itervalues(latent_vars))
  q_vars = [v for v in var_list
            if len(get_descendants(tf.convert_to_tensor(v), q_rvs)) != 0]
  q_grads = tf.gradients(
      -(tf.reduce_mean(q_log_prob * tf.stop_gradient(losses)) - reg_penalty),
      q_vars)
  p_vars = [v for v in var_list if v not in q_vars]
  p_grads = tf.gradients(loss, p_vars)
  grads_and_vars = list(zip(q_grads, q_vars)) + list(zip(p_grads, p_vars))
  return loss, grads_and_vars


def build_score_kl_loss_and_gradients(
    latent_vars, data, var_list, scale, n_samples, kl_scaling, summary_key):
  """Build loss function and gradients based on the score function
  estimator [@paisley2012variational].

  It assumes the KL is analytic.

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_lik = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value
      q_log_prob[s] += tf.reduce_sum(
          scale.get(z, 1.0) *
          qz_copy.log_prob(tf.stop_gradient(dict_swap[z])))

    for x in six.iterkeys(data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_lik[s] += tf.reduce_sum(
            scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

  p_log_lik = tf.stack(p_log_lik)
  q_log_prob = tf.stack(q_log_prob)

  kl_penalty = tf.reduce_sum([
      tf.reduce_sum(kl_scaling.get(z, 1.0) * kl_divergence(qz, z))
      for z, qz in six.iteritems(latent_vars)])

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())

  if summary_key is not None:
    tf.summary.scalar("loss/p_log_lik", tf.reduce_mean(p_log_lik),
                      collections=[summary_key])
    tf.summary.scalar("loss/kl_penalty", kl_penalty,
                      collections=[summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[summary_key])

  loss = -(tf.reduce_mean(p_log_lik) - kl_penalty - reg_penalty)

  q_rvs = list(six.itervalues(latent_vars))
  q_vars = [v for v in var_list
            if len(get_descendants(tf.convert_to_tensor(v), q_rvs)) != 0]
  q_grads = tf.gradients(
      -(tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_lik)) - kl_penalty -
          reg_penalty),
      q_vars)
  p_vars = [v for v in var_list if v not in q_vars]
  p_grads = tf.gradients(loss, p_vars)
  grads_and_vars = list(zip(q_grads, q_vars)) + list(zip(p_grads, p_vars))
  return loss, grads_and_vars


def build_score_entropy_loss_and_gradients(
    latent_vars, data, var_list, scale, n_samples, summary_key):
  """Build loss function and gradients based on the score function
  estimator [@paisley2012variational].

  It assumes the entropy is analytic.

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value
      q_log_prob[s] += tf.reduce_sum(
          scale.get(z, 1.0) *
          qz_copy.log_prob(tf.stop_gradient(dict_swap[z])))

    for z in six.iterkeys(latent_vars):
      z_copy = copy(z, dict_swap, scope=scope)
      p_log_prob[s] += tf.reduce_sum(
          scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

  p_log_prob = tf.stack(p_log_prob)
  q_log_prob = tf.stack(q_log_prob)

  q_entropy = tf.reduce_sum([
      tf.reduce_sum(qz.entropy())
      for z, qz in six.iteritems(latent_vars)])

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())

  if summary_key is not None:
    tf.summary.scalar("loss/p_log_prob", tf.reduce_mean(p_log_prob),
                      collections=[summary_key])
    tf.summary.scalar("loss/q_log_prob", tf.reduce_mean(q_log_prob),
                      collections=[summary_key])
    tf.summary.scalar("loss/q_entropy", q_entropy,
                      collections=[summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[summary_key])

  loss = -(tf.reduce_mean(p_log_prob) + q_entropy - reg_penalty)

  q_rvs = list(six.itervalues(latent_vars))
  q_vars = [v for v in var_list
            if len(get_descendants(tf.convert_to_tensor(v), q_rvs)) != 0]
  q_grads = tf.gradients(
      -(tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_prob)) +
          q_entropy - reg_penalty),
      q_vars)
  p_vars = [v for v in var_list if v not in q_vars]
  p_grads = tf.gradients(loss, p_vars)
  grads_and_vars = list(zip(q_grads, q_vars)) + list(zip(p_grads, p_vars))
  return loss, grads_and_vars


def build_score_rb_loss_and_gradients(
    latent_vars, data, var_list, scale, n_samples, summary_key):
  """Build loss function and gradients based on the score function
  estimator [@paisley2012variational] and Rao-Blackwellization
  [@ranganath2014black].

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling and Rao-Blackwellization.
  """
  # Build tensors for loss and gradient calculations. There is one set
  # for each sample from the variational distribution.
  p_log_probs = [{}] * n_samples
  q_log_probs = [{}] * n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value
      q_log_probs[s][qz] = tf.reduce_sum(
          scale.get(z, 1.0) *
          qz_copy.log_prob(tf.stop_gradient(dict_swap[z])))

    for z in six.iterkeys(latent_vars):
      z_copy = copy(z, dict_swap, scope=scope)
      p_log_probs[s][z] = tf.reduce_sum(
          scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_probs[s][x] = tf.reduce_sum(
            scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

  # Take gradients of Rao-Blackwellized loss for each variational parameter.
  p_rvs = list(six.iterkeys(latent_vars)) + \
      [x for x in six.iterkeys(data) if isinstance(x, RandomVariable)]
  q_rvs = list(six.itervalues(latent_vars))
  reverse_latent_vars = {v: k for k, v in six.iteritems(latent_vars)}
  grads = []
  grads_vars = []
  for var in var_list:
    # Get all variational factors depending on the parameter.
    descendants = get_descendants(tf.convert_to_tensor(var), q_rvs)
    if len(descendants) == 0:
      continue  # skip if not a variational parameter
    # Get p and q's Markov blanket wrt these latent variables.
    var_p_rvs = set()
    for qz in descendants:
      z = reverse_latent_vars[qz]
      var_p_rvs.update(z.get_blanket(p_rvs) + [z])

    var_q_rvs = set()
    for qz in descendants:
      var_q_rvs.update(qz.get_blanket(q_rvs) + [qz])

    pi_log_prob = [0.0] * n_samples
    qi_log_prob = [0.0] * n_samples
    for s in range(n_samples):
      pi_log_prob[s] = tf.reduce_sum([p_log_probs[s][rv] for rv in var_p_rvs])
      qi_log_prob[s] = tf.reduce_sum([q_log_probs[s][rv] for rv in var_q_rvs])

    pi_log_prob = tf.stack(pi_log_prob)
    qi_log_prob = tf.stack(qi_log_prob)
    grad = tf.gradients(
        -tf.reduce_mean(qi_log_prob *
                        tf.stop_gradient(pi_log_prob - qi_log_prob)) +
        tf.reduce_sum(tf.losses.get_regularization_losses()),
        var)
    grads.extend(grad)
    grads_vars.append(var)

  # Take gradients of total loss function for model parameters.
  loss = -(tf.reduce_mean([tf.reduce_sum(list(six.itervalues(p_log_prob)))
                           for p_log_prob in p_log_probs]) -
           tf.reduce_mean([tf.reduce_sum(list(six.itervalues(q_log_prob)))
                           for q_log_prob in q_log_probs]) -
           tf.reduce_sum(tf.losses.get_regularization_losses()))
  model_vars = [v for v in var_list if v not in grads_vars]
  model_grads = tf.gradients(loss, model_vars)
  grads.extend(model_grads)
  grads_vars.extend(model_vars)
  grads_and_vars = list(zip(grads, grads_vars))
  return loss, grads_and_vars
