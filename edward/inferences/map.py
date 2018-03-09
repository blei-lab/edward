from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.util import (
    call_with_intercept, call_with_trace, toposort)

try:
  from tensorflow.contrib.distributions import bijectors
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_variational +
          doc.arg_align_latent +
          doc.arg_align_data +
          doc.arg_scale +
          doc.arg_auto_transform +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_loss,
    notes_regularization_losses=doc.notes_regularization_losses)
def map(model, variational, align_latent, align_data,
        scale=lambda name: 1.0, auto_transform=True, collections=None,
        *args, **kwargs):
  """Maximum a posteriori.

  This function implements gradient-based optimization to solve the
  optimization problem,

  $\min_{z} - p(z \mid x).$

  This is equivalent to using a `PointMass` variational distribution
  and minimizing the unnormalized objective,

  $- \mathbb{E}_{q(z; \lambda)} [ \log p(x, z) ].$

  Args:
  @{args}

  Returns:
  @{returns}

  #### Notes

  This function is currently restricted to optimization over
  differentiable latent variables. For example, it does not solve
  discrete optimization.

  Probabilistic programs may have random variables which vary across
  executions. The algorithm returns calculations following one
  execution of the model and variational programs.

  This function also minimizes the loss with respect to any model
  parameters $p(z \mid x; \\theta)$.

  In conditional inference, we infer $z$ in $p(z, \\beta
  \mid x)$ while fixing inference over $\\beta$ using another
  distribution $q(\\beta)$. `MAP` optimizes
  $\mathbb{E}_{q(\\beta)} [ \log p(x, z, \\beta) ]$, leveraging
  a single Monte Carlo sample, $\log p(x, z, \\beta^*)$, where
  $\\beta^* \sim q(\\beta)$. This is a lower bound to the
  marginal density $\log p(x, z)$, and it is exact if
  $q(\\beta) = p(\\beta \mid x)$ (up to stochasticity).

  @{notes_regularization_losses}

  #### Examples

  Most explicitly, this function is specified via a variational
  program over pointmasses.

  ```python
  def variational():
    qpi = PointMass(params=to_simplex(tf.Variable(tf.zeros(K-1))),
                    name="qpi")
    qmu = PointMass(params=tf.Variable(tf.zeros(K*D)),
                    name="qmu")
    qsigma = PointMass(params=tf.nn.softplus(tf.Variable(tf.zeros(K*D))),
                       name="qsigma")
    return qpi, qmu, qsigma

  loss = ed.map(..., variational, ...)
  ```

  We also automate the specification of `PointMass` distributions
  so you don't pass in `variational`. (TODO not implemented yet.)

  Note that for this function to optimize over latent variables with
  constrained continuous support, the point mass must be constrained
  to have the same support while its free parameters are
  unconstrained; see, e.g., `qsigma` above. This is different than
  performing MAP on the unconstrained space: in general, the MAP of
  the transform is not the transform of the MAP.
  """
  q_trace = call_with_trace(variational, *args, **kwargs)
  x = call_with_intercept(model, q_trace, align_data, align_latent,
                          *args, **kwargs)
  p_log_prob = 0.0
  for rv in toposort(x):
    if align_latent(rv.name) is not None or align_data(rv.name) is not None:
      scale_factor = scale(rv.name)
      p_log_prob += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if collections is not None:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=collections)
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=collections)

  loss = -p_log_prob + reg_penalty
  return loss
