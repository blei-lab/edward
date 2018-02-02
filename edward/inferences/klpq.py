from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.util import make_intercept
from edward.models.core import trace

try:
  from edward.models import Normal
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_variational +
          doc.arg_align_latent +
          doc.arg_align_data +
          doc.arg_scale +
          doc.arg_n_samples +
          doc.arg_auto_transform +
          doc.arg_collections +
          doc.arg_args_kwargs),
    returns=doc.return_loss_surrogate_loss,
    notes_model_parameters=doc.notes_model_parameters,
    notes_conditional_inference=doc.notes_conditional_inference_samples,
    notes_regularization_losses=doc.notes_regularization_losses)
def klpq(model, variational, align_latent, align_data,
         scale=lambda name: 1.0, n_samples=1, auto_transform=True,
         collections=None, *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( p(z \mid x) \| q(z) )
    = \mathbb{E}_{p(z \mid x)} [ \log p(z \mid x) - \log q(z; \lambda) ]$.

  To perform the optimization, this function uses a technique from
  adaptive importance sampling [@oh1992adaptive].

  The loss function can be estimated up to a constant as

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

  Args:
  @{args}

  Returns:
  @{returns}

  #### Notes

  Probabilistic programs may have random variables which vary across
  executions. The algorithm returns calculations following `n_samples`
  executions of the model and variational programs.

  @{notes_model_parameters}

  @{notes_conditional_inference}

  @{notes_regularization_losses}

  #### Examples

  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")

  def variational():
    qmu = Normal(loc=tf.get_variable("loc", []),
                 scale=tf.nn.softplus(tf.get_variable("shape", [])),
                 name="qmu")

  loss, surrogate_loss = ed.klpq(
      model, variational,
      align_latent=lambda name: "qmu" if name == "mu" else None,
      align_data=lambda name: "x" if name == "x" else None,
      x=x_data)
  ```
  """
  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  for s in range(n_samples):
    posterior_trace = trace(variational, *args, **kwargs)
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    model_trace = trace(model, intercept=intercept, *args, **kwargs)

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
