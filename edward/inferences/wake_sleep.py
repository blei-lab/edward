from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.util import make_intercept
from edward.models.core import trace


@doc.set_doc(
    args_part_one=(doc.arg_model +
                   doc.arg_variational +
                   doc.arg_align_latent +
                   doc.arg_align_data +
                   doc.arg_scale +
                   doc.arg_n_samples)[:-1],
    args_part_two=(doc.arg_auto_transform +
                   doc.arg_collections +
                   doc.arg_args_kwargs)[:-1],
    notes_conditional_inference=doc.notes_conditional_inference_samples,
    notes_regularization_losses=doc.notes_regularization_losses)
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

  Args:
  @{args_part_one}
    phase_q: str.
      Phase for updating parameters of q. If 'sleep', update using
      a sample from p. If 'wake', update using a sample from q.
      (Unlike reparameterization gradients, the sample is held
      fixed.)
  @{args_part_two}

  Returns:
    Pair of scalar tf.Tensors, representing losses for training p
    and q respectively.

  #### Notes

  Probabilistic programs may have random variables which vary across
  executions. The algorithm returns calculations following `n_samples`
  executions of the model and variational programs.

  @{notes_conditional_inference}

  @{notes_regularization_losses}

  #### Examples

  ```python
  def model():
    z = Normal(loc=0.0, scale=1.0, sample_shape=[256, 25], name="z")
    net = tf.layers.dense(z, 512, activation=tf.nn.relu)
    net = tf.layers.dense(net, 28 * 28, activation=None)
    x = Normal(loc=net, scale=1.0, name="x")

  def variational(x):
    net = tf.layers.dense(x, 25 * 2)
    qz = Normal(loc=net[:, :25],
                scale=tf.nn.softplus(net[:, 25:]),
                name="qz")

  loss_p, loss_q = ed.wake_sleep(
      model, variational,
      align_latent=lambda name: "qz" if name == "z" else None,
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
      if align_data(name) is not None or align_latent(name) is not None:
        p_log_prob[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      if phase_q != 'sleep' and align_latent(name) is not None:
        # If not sleep phase, compute log q(z).
        qz = posterior_trace[align_latent(name)].value
        q_log_prob[s] += tf.reduce_sum(
            scale_factor * qz.log_prob(tf.stop_gradient(qz.value)))

    if phase_q == 'sleep':
      model_trace = trace(model, *args, **kwargs)
      intercept = _make_sleep_intercept(
          model_trace, align_data, align_latent, args, kwargs)
      posterior_trace = trace(variational, intercept=intercept, *args, **kwargs)

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
