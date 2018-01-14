from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import (
    call_function_up_to_args, make_intercept)
from edward.models.core import Trace


def wake_sleep(model, variational, align_latent, align_data,
               scale=lambda name: 1.0, n_samples=1, phase_q='sleep',
               auto_transform=True, collections=None, *args, **kwargs):
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
  """
  Args:
    n_samples: int, optional.
      Number of samples for calculating stochastic gradients during
      wake and sleep phases.
    phase_q: str, optional.
      Phase for updating parameters of q. If 'sleep', update using
      a sample from p. If 'wake', update using a sample from q.
      (Unlike reparameterization gradients, the sample is held
      fixed.)
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
      if align_data(name) is not None or align_latent(name) is not None:
        p_log_prob[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      if phase_q != 'sleep' and align_latent(name) is not None:
        # If not sleep phase, compute log q(z).
        qz = posterior_trace[align_latent(name)].value
        q_log_prob[s] += tf.reduce_sum(
            scale_factor * qz.log_prob(tf.stop_gradient(qz.value)))

    if phase_q == 'sleep':
      with Trace() as model_trace:
        call_function_up_to_args(model, *args, **kwargs)
      intercept = _make_sleep_intercept(
          model_trace, align_data, align_latent, args, kwargs)
      with Trace(intercept=intercept) as posterior_trace:
        call_function_up_to_args(variational, *args, **kwargs)

      # Build dictionary to return scale factor for a posterior
      # variable via its corresponding prior. The implementation is
      # naive.
      scale_posterior = {}
      for name, node in six.iteritems(model_trace):
        rv = node.value
        if align_latent(name) is not None:
          qz = posterior_trace[align_latent(name)].value
          scale_posterior[qz] = rv

      for name, node in six.iteritems(posterior_trace):
        rv = node.value
        scale_factor = scale_posterior[rv]
        q_log_prob[s] += tf.reduce_sum(
            scale_factor * rv.log_prob(tf.stop_gradient(rv.value)))

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

  loss_p = -p_log_prob + reg_penalty
  loss_q = -q_log_prob + reg_penalty
  return loss_p, loss_q


def _make_sleep_intercept(trace, align_data, align_latent, args, kwargs):
  def _intercept(f, *fargs, **fkwargs):
    """Set variational distribution's sample value to prior's."""
    name = fkwargs.get('name', None)
    z = trace[align_latent(name)].value
    fkwargs['value'] = z.value
    return f(*fargs, **fkwargs)
  return _intercept
