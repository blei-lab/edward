from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import (
    call_function_up_to_args, make_intercept)
from edward.models.core import Trace

try:
  from edward.models import Normal
  from tensorflow.contrib.distributions import kl_divergence
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))

tfd = tf.contrib.distributions


def klqp(model, variational, align_latent, align_data,
         scale=lambda name: 1.0, n_samples=1, kl_scaling=lambda name: 1.0,
         auto_transform=True, collections=None, *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective by automatically selecting from a
  variety of black box inference techniques.

  Args:
    model: function whose inputs are a subset of `args` (e.g., for
      discriminative). Output is not used.
      TODO auto_transform docstring
      Collection of random variables to perform inference on.
      If list, each random variable will be implictly optimized using
      a `Normal` random variable that is defined internally with a
      free parameter per location and scale and is initialized using
      standard normal draws. The random variables to approximate must
      be continuous.
    variational: function whose inputs are a subset of `args` (e.g.,
      for amortized). Output is not used.
    align_latent: function of string, aligning `model` latent
      variables with `variational`. It takes a model variable's name
      as input and returns a string, indexing `variational`'s trace;
      else identity.
    align_data: function of string, aligning `model` observed
      variables with data. It takes a model variable's name as input
      and returns an integer, indexing `args`; else identity.
    scale: function of string, aligning `model` observed
      variables with scale factors. It takes a model variable's name
      as input and returns a scale factor; else 1.0. The scale
      factor's shape must be broadcastable; it is multiplied
      element-wise to the random variable. For example, this is useful
      for mini-batch scaling when inferring global variables, or
      applying masks on a random variable.
    n_samples: int, optional.
      Number of samples from variational model for calculating
      stochastic gradients.
    kl_scaling: function of string, aligning `model` latent
      variables with KL scale factors. This provides option to scale
      terms when using ELBO with KL divergence. If the KL divergence
      terms are

      $\\alpha_p \mathbb{E}_{q(z\mid x, \lambda)} [
            \log q(z\mid x, \lambda) - \log p(z)],$

      then pass {$p(z)$: $\\alpha_p$} as `kl_scaling`,
      where $\\alpha_p$ is a tensor. Its shape must be broadcastable;
      it is multiplied element-wise to the batchwise KL terms.
    args: data inputs. It is passed at compile-time in Graph
      mode or runtime in Eager mode.

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

  This class minimizes the objective using the score function gradient
  and Rao-Blackwellization [@ranganath2014black].

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling and Rao-Blackwellization.

  The implementation takes the surrogate loss approach. See
  @schulman2015stochastic; @ruiz2016generalized; @ritchie2016deep.

  #### Notes

  Current Rao-Blackwellization is limited to Rao-Blackwellizing across
  stochastic nodes in the computation graph. It does not
  Rao-Blackwellize within a node such as when a node represents
  multiple random variables via non-scalar batch shape.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  # TODO control variates
  # + baseline, learnable baseline
  # + Ruiz+ 2016
  # + Tucker+ 2017; Cremer+ 2017
  # + Miller+ 2017
  # TODO analytic stuff
  # + Roeder+ 2017
  p_log_prob = [None] * n_samples
  q_log_prob = [None] * n_samples
  surrogate_loss = [None] * n_samples
  kl_penalty = 0.0
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    # Collect key-value pairs of (rv, rv's (scaled) log prob).
    p_dict = {}
    q_dict = {}
    inverse_align_latent = {}
    for name, node in six.iteritems(model_trace):
      rv = node.value
      scale_factor = scale(name)
      if align_data(name) is not None:
        p_dict[rv] = tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      if align_latent(name) is not None:
        qz = posterior_trace[align_latent(name)].value
        # For pairs with analytic KL, accumulate KL divergences for
        # first iteration in loop.
        if isinstance(rv, Normal) and isinstance(qz, Normal):
          if s == 0:
            kl_penalty += tf.reduce_sum(
                kl_scaling(name) * kl_divergence(qz, rv))
        else:
          p_dict[rv] = tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
          q_dict[qz] = tf.reduce_sum(scale_factor * qz.log_prob(qz.value))
          inverse_align_latent[qz] = rv

    # Build surrogate loss.
    scaled_q_log_prob = 0.0
    for qz, log_prob in six.iteritems(q_dict):
      if qz.reparameterization_type == tfd.FULLY_REPARAMETERIZED:
        scale_factor = 1.0
      else:
        scale_factor = 0.0
        for rv in qz.get_blanket(q_rvs) + [qz]:
          scale_factor += q_dict[rv]
          scale_factor -= p_dict[inverse_align_latent[qz]]
      scaled_q_log_prob += scale_factor * log_prob

    p_log_prob_s = tf.reduce_sum(list(six.itervalues(p_dict)))
    p_log_prob[s] = p_log_prob_s
    q_log_prob[s] = tf.reduce_sum(list(six.itervalues(q_dict)))
    surrogate_loss[s] = scaled_q_log_prob - p_log_prob_s

  p_log_prob = tf.reduce_mean(p_log_prob)
  q_log_prob = tf.reduce_mean(q_log_prob)
  surrogate_loss = tf.reduce_mean(surrogate_loss) + kl_penalty

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  surrogate_loss += reg_penalty

  if collections is not None:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=collections)
    tf.summary.scalar("loss/q_log_prob", q_log_prob,
                      collections=collections)
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=collections)

  loss = q_log_prob - p_log_prob + kl_penalty + reg_penalty
  return loss, surrogate_loss


def klqp_reparameterization(model, variational, align_latent, align_data,
                            scale=lambda name: 1.0, n_samples=1,
                            auto_transform=True, collections=None,
                            *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the reparameterization
  gradient.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  Build loss function equal to KL(q||p) up to a constant. Its
  automatic differentiation is a stochastic gradient of

  $-\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

  based on the reparameterization trick [@kingma2014auto].

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.

  Note if user defines constrained posterior, then auto_transform
  can do inference on real-valued; then test time user can use
  constrained. If user defines unconstrained posterior, then how to
  work with constrained at test time? For now, user must manually
  write the bijectors according to transform.
  """
  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    for name, node in six.iteritems(model_trace):
      rv = node.value
      scale_factor = scale(name)
      if align_latent(name) is not None or align_data(name) is not None:
        p_log_prob[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      if align_latent(name) is not None:
        qz = posterior_trace[align_latent(name)].value
        q_log_prob[s] += tf.reduce_sum(scale_factor * qz.log_prob(qz.value))

  p_log_prob = tf.reduce_mean(p_log_prob)
  q_log_prob = tf.reduce_mean(q_log_prob)
  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if collections is not None:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=collections)
    tf.summary.scalar("loss/q_log_prob", q_log_prob,
                      collections=collections)
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=collections)
  loss = q_log_prob - p_log_prob + reg_penalty
  return loss


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_variational +
          doc.arg_align_latent +
          doc.arg_align_data +
          doc.arg_scale +
          doc.arg_n_samples +
          doc.arg_kl_scaling +
          doc.arg_auto_transform +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_loss,
    notes_model_parameters=doc.notes_model_parameters,
    notes_conditional_inference=doc.notes_conditional_inference_samples,
    notes_regularization_losses=doc.notes_regularization_losses)
def klqp_reparameterization_kl(model, variational, align_latent, align_data,
                               scale=lambda name: 1.0, n_samples=1,
                               kl_scaling=lambda name: 1.0,
                               auto_transform=True, collections=None,
                               *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the reparameterization
  gradient and an analytic KL term.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  Build loss function. Its automatic differentiation
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
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    for name, node in six.iteritems(model_trace):
      if align_data(name) is not None:
        rv = node.value
        scale_factor = scale(name)
        p_log_lik[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))

  p_log_lik = tf.reduce_mean(p_log_lik)

  kl_penalty = 0.0
  for name, node in six.iteritems(model_trace):
    if align_latent(name) is not None:
      rv = node.value
      qz = posterior_trace[align_latent(name)].value
      kl_penalty += tf.reduce_sum(kl_scaling(name) * kl_divergence(qz, rv))

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if collections is not None:
    tf.summary.scalar("loss/p_log_lik", p_log_lik,
                      collections=collections)
    tf.summary.scalar("loss/kl_penalty", kl_penalty,
                      collections=collections)
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=collections)
  loss = -p_log_lik + kl_penalty + reg_penalty
  return loss


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_variational +
          doc.arg_align_latent +
          doc.arg_align_data +
          doc.arg_scale +
          doc.arg_n_samples +
          doc.arg_auto_transform +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_loss_surrogate_loss,
    notes_model_parameters=doc.notes_model_parameters,
    notes_conditional_inference=doc.notes_conditional_inference_samples,
    notes_regularization_losses=doc.notes_regularization_losses)
def klqp_score(model, variational, align_latent, align_data,
               scale=lambda name: 1.0, n_samples=1, auto_transform=True,
               collections=None, *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the score function
  gradient.

  Build loss function equal to KL(q||p) up to a constant. It
  returns an surrogate loss function whose automatic differentiation
  is based on the score function estimator [@paisley2012variational].

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    for name, node in six.iteritems(model_trace):
      rv = node.value
      scale_factor = scale(name)
      if align_latent(name) is not None or align_data(name) is not None:
        p_log_prob[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      if align_latent(name) is not None:
        qz = posterior_trace[align_latent(name)].value
        q_log_prob[s] += tf.reduce_sum(
            scale_factor * qz.log_prob(tf.stop_gradient(qz.value)))

  p_log_prob = tf.stack(p_log_prob)
  q_log_prob = tf.stack(q_log_prob)
  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if collections is not None:
    tf.summary.scalar("loss/p_log_prob", tf.reduce_mean(p_log_prob),
                      collections=collections)
    tf.summary.scalar("loss/q_log_prob", tf.reduce_mean(q_log_prob),
                      collections=collections)
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=collections)
  losses = q_log_prob - p_log_prob
  loss = tf.reduce_mean(losses) + reg_penalty
  surrogate_loss = (tf.reduce_mean(q_log_prob * tf.stop_gradient(losses)) +
                    reg_penalty)
  return loss, surrogate_loss
