from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import (
    call_function_up_to_args, make_intercept)
from edward.models.core import Trace


def klqp_implicit(model, variational, discriminator, align_latent,
                  align_data, align_latent_global=lambda name: name,
                  ratio_loss='log', scale=lambda name: 1.0,
                  auto_transform=True, collections=None, *args, **kwargs):
  """Variational inference with implicit probabilistic models
  [@tran2017deep].

  It minimizes the KL divergence

  $\\text{KL}( q(z, \\beta; \lambda) \| p(z, \\beta \mid x) ),$

  where $z$ are local variables associated to a data point and
  $\\beta$ are global variables shared across data points.

  Global latent variables require `log_prob()` and need to return a
  random sample when fetched from the graph. Local latent variables
  and observed variables require only a random sample when fetched
  from the graph. (This is true for both $p$ and $q$.)

  All variational factors must be reparameterizable: each of the
  random variables (`rv`) satisfies `rv.is_reparameterized` and
  `rv.is_continuous`.

  #### Notes

  Unlike `GANInference`, `discriminator` takes dict's as input,
  and must subset to the appropriate values through lexical scoping
  from the previously defined model and latent variables. This is
  necessary as the discriminator can take an arbitrary set of data,
  latent, and global variables.

  Note the type for `discriminator`'s output changes when one
  passes in the `scale` argument to `initialize()`.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  """Create an inference algorithm.

  Args:
    discriminator: function.
      Function (with parameters). Unlike `GANInference`, it is
      interpreted as a ratio estimator rather than a discriminator.
      It takes three arguments: a data dict, local latent variable
      dict, and global latent variable dict. As with GAN
      discriminators, it can take a batch of data points and local
      variables, of size $M$, and output a vector of length
      $M$.
    global_vars: dict of RandomVariable to RandomVariable, optional.
      Identifying which variables in `latent_vars` are global
      variables, shared across data points. These will not be
      encompassed in the ratio estimation problem, and will be
      estimated with tractable variational approximations.
  """
  """Initialize inference algorithm. It initializes hyperparameters
  and builds ops for the algorithm's computation graph.

  Args:
    ratio_loss: str or fn, optional.
      Loss function minimized to get the ratio estimator. 'log' or 'hinge'.
      Alternatively, one can pass in a function of two inputs,
      `psamples` and `qsamples`, and output a point-wise value
      with shape matching the shapes of the two inputs.
  """
  """Build loss function

  $-\Big(\mathbb{E}_{q(\\beta)} [\log p(\\beta) - \log q(\\beta) ] +
      \sum_{n=1}^N \mathbb{E}_{q(\\beta)q(z_n\mid\\beta)} [
          r^*(x_n, z_n, \\beta) ] \Big).$

  We minimize it with respect to parameterized variational
  families $q(z, \\beta; \lambda)$.

  $r^*(x_n, z_n, \\beta)$ is a function of a single data point
  $x_n$, single local variable $z_n$, and all global
  variables $\\beta$. It is equal to the log-ratio

  $\log p(x_n, z_n\mid \\beta) - \log q(x_n, z_n\mid \\beta),$

  where $q(x_n)$ is the empirical data distribution. Rather
  than explicit calculation, $r^*(x, z, \\beta)$ is the
  solution to a ratio estimation problem, minimizing the specified
  `ratio_loss`.

  Gradients are taken using the reparameterization trick
  [@kingma2014auto].

  #### Notes

  This also includes model parameters $p(x, z, \\beta; \\theta)$
  and variational distributions with inference networks
  $q(z\mid x)$.

  There are a bunch of extensions we could easily do in this
  implementation:

  + further factorizations can be used to better leverage the
    graph structure for more complicated models;
  + score function gradients for global variables;
  + use more samples; this would require the `copy()` utility
    function for q's as well, and an additional loop. we opt not to
    because it complicates the code;
  + analytic KL/swapping out the penalty term for the globals.

  align_latent aligns all global and local latents;
  align_global_latent only aligns global latents.
  """
  if callable(ratio_loss):
    ratio_loss = ratio_loss
  elif ratio_loss == 'log':
    ratio_loss = _log_loss
  elif ratio_loss == 'hinge':
    ratio_loss = _hinge_loss
  else:
    raise ValueError('Ratio loss not found:', ratio_loss)

  with Trace() as posterior_trace:
    call_function_up_to_args(variational, *args, **kwargs)
  global_intercept = make_intercept(
      posterior_trace, align_data, align_latent_global, args, kwargs)
  with Trace(intercept=global_intercept) as model_trace:
    # Intercept model's global latent variables and set to posterior
    # samples (but not its locals).
    call_function_up_to_args(model, *args, **kwargs)

  # Collect tensors used in calculation of losses.
  pbeta_log_prob = 0.0
  qbeta_log_prob = 0.0
  qbeta_sample = {}
  pz_sample = {}
  qz_sample = {}
  x_psample = {}
  x_qsample = {}
  for name, node in six.iteritems(model_trace):
    # Calculate log p(beta') and log q(beta').
    if align_latent_global(name) is not None:
      pbeta = node.value
      qbeta = posterior_trace[align_latent_global(name)].value
      scale_factor = scale(name)
      pbeta_log_prob += tf.reduce_sum(
          scale_factor * pbeta.log_prob(pbeta.value))
      qbeta_log_prob += tf.reduce_sum(
          scale_factor * qbeta.log_prob(qbeta.value))
      qbeta_sample[name] = qbeta.value
    else:
      # TODO This assumes implicit variables are tf.Tensors existing
      # on the Trace stack.
      if align_latent(name) is not None:
        pz = node.value
        qz = posterior_trace[align_latent(Name)].value
        pz_sample[name] = pz
        qz_sample[name] = qz
      else:
        key = align_data(name)
        if isinstance(key, int):
          data_node = args[key]
        elif kwargs.get(key, None) is not None:
          data_node = kwargs.get(key)
        px = node.value
        qx = data_node.value
        x_psample[name] = px
        x_qsample[name] = qx

  # Collect x' ~ p(x | z', beta') and x' ~ q(x).
  with tf.variable_scope("Disc"):
    # TODO For now, this assumes the discriminator automagically knows
    # how to index the dictionaries and computes some forward pass on
    # them (which can vary across executions). Dictionaries should be
    # improved to be more idiomatic.
    r_psample = discriminator(x_psample, pz_sample, qbeta_sample)

  with tf.variable_scope("Disc", reuse=True):
    r_qsample = discriminator(x_qsample, qz_sample, qbeta_sample)

  # Form ratio loss and ratio estimator.
  loss_d = 0.0
  scaled_ratio = 0.0
  for key, value in six.iteritems(r_qsample):
    loss_d += tf.reduce_mean(ratio_loss(r_psample[key], value))
    scaled_ratio += tf.reduce_sum(scale(key) * value)

  reg_terms_d = tf.losses.get_regularization_losses(scope="Disc")
  reg_terms_all = tf.losses.get_regularization_losses()
  reg_terms = [r for r in reg_terms_all if r not in reg_terms_d]

  # Form variational objective.
  loss = (qbeta_log_prob - pbeta_log_prob - scaled_ratio +
          tf.reduce_sum(reg_terms))
  loss_d = loss_d + tf.reduce_sum(reg_terms_d)
  return loss, loss_d


def _log_loss(psample, qsample):
  """Point-wise log loss."""
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(psample), logits=psample) + \
      tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.zeros_like(qsample), logits=qsample)
  return loss


def _hinge_loss(psample, qsample):
  """Point-wise hinge loss."""
  loss = tf.nn.relu(1.0 - psample) + tf.nn.relu(1.0 + qsample)
  return loss
