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
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


def klpq(model, variational, align_latent, align_data,
         scale=lambda name: 1.0, n_samples=1, auto_transform=True,
         collections=None, *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( p(z \mid x) \| q(z) ).$

  To perform the optimization, this class uses a technique from
  adaptive importance sampling [@oh1992adaptive].

  #### Notes

  `KLpq` also optimizes any model parameters $p(z\mid x;
  \\theta)$. It does this by variational EM, maximizing

  $\mathbb{E}_{p(z \mid x; \lambda)} [ \log p(x, z; \\theta) ]$

  with respect to $\\theta$.

  In conditional inference, we infer $z` in $p(z, \\beta
  \mid x)$ while fixing inference over $\\beta$ using another
  distribution $q(\\beta)$. During gradient calculation, instead
  of using the model's density

  $\log p(x, z^{(s)}), z^{(s)} \sim q(z; \lambda),$

  for each sample $s=1,\ldots,S$, `KLpq` uses

  $\log p(x, z^{(s)}, \\beta^{(s)}),$

  where $z^{(s)} \sim q(z; \lambda)$ and$\\beta^{(s)}
  \sim q(\\beta)$.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  """Create an inference algorithm.

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
  """
  """Build loss function

  $\\text{KL}( p(z \mid x) \| q(z) )
    = \mathbb{E}_{p(z \mid x)} [ \log p(z \mid x) - \log q(z; \lambda) ]$

  and stochastic gradients based on importance sampling.

  The loss function can be estimated as

  $\sum_{s=1}^S [
    w_{\\text{norm}}(z^s; \lambda) (\log p(x, z^s) - \log q(z^s; \lambda) ],$

  where for $z^s \sim q(z; \lambda)$,

  $w_{\\text{norm}}(z^s; \lambda) =
        w(z^s; \lambda) / \sum_{s=1}^S w(z^s; \lambda)$

  normalizes the importance weights, $w(z^s; \lambda) = p(x,
  z^s) / q(z^s; \lambda)$.

  This provides a gradient,

  $- \sum_{s=1}^S [
    w_{\\text{norm}}(z^s; \lambda) \\nabla_{\lambda} \log q(z^s; \lambda) ].$
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
        p_log_prob[s] += tf.reduce_sum(
            scale_factor * rv.log_prob(tf.stop_gradient(rv.value)))
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

  log_w = p_log_prob - tf.stop_gradient(q_log_prob)
  log_w_norm = log_w - tf.reduce_logsumexp(log_w)
  w_norm = tf.exp(log_w_norm)
  loss = -tf.reduce_sum(w_norm * log_w) + reg_penalty
  # Model parameter gradients will backprop into loss. Variational
  # parameter gradients will backprop into reg_penalty and last term.
  surrogate_loss = loss + tf.reduce_sum(q_log_prob * tf.stop_gradient(w_norm))
  return loss, surrogate_loss
