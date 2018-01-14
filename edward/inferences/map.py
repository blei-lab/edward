from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import (
    call_function_up_to_args, make_intercept)
from edward.models.core import Trace

try:
  from tensorflow.contrib.distributions import bijectors
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


def map(model, variational, align_latent, align_data,
        scale=lambda name: 1.0, auto_transform=True, collections=None,
        *args, **kwargs):
  """Maximum a posteriori.

  This class implements gradient-based optimization to solve the
  optimization problem,

  $\min_{z} - p(z \mid x).$

  This is equivalent to using a `PointMass` variational distribution
  and minimizing the unnormalized objective,

  $- \mathbb{E}_{q(z; \lambda)} [ \log p(x, z) ].$

  #### Notes

  This class is currently restricted to optimization over
  differentiable latent variables. For example, it does not solve
  discrete optimization.

  This class also minimizes the loss with respect to any model
  parameters $p(z \mid x; \\theta)$.

  In conditional inference, we infer $z$ in $p(z, \\beta
  \mid x)$ while fixing inference over $\\beta$ using another
  distribution $q(\\beta)$. `MAP` optimizes
  $\mathbb{E}_{q(\\beta)} [ \log p(x, z, \\beta) ]$, leveraging
  a single Monte Carlo sample, $\log p(x, z, \\beta^*)$, where
  $\\beta^* \sim q(\\beta)$. This is a lower bound to the
  marginal density $\log p(x, z)$, and it is exact if
  $q(\\beta) = p(\\beta \mid x)$ (up to stochasticity).

  #### Examples

  Most explicitly, `MAP` is specified via a dictionary:

  ```python
  qpi = PointMass(params=ed.to_simplex(tf.Variable(tf.zeros(K-1))))
  qmu = PointMass(params=tf.Variable(tf.zeros(K*D)))
  qsigma = PointMass(params=tf.nn.softplus(tf.Variable(tf.zeros(K*D))))
  ed.MAP({pi: qpi, mu: qmu, sigma: qsigma}, data)
  ```

  We also automate the specification of `PointMass` distributions,
  so one can pass in a list of latent variables instead:

  ```python
  ed.MAP([beta], data)
  ed.MAP([pi, mu, sigma], data)
  ```

  Note that for `MAP` to optimize over latent variables with
  constrained continuous support, the point mass must be constrained
  to have the same support while its free parameters are
  unconstrained; see, e.g., `qsigma` above. This is different than
  performing MAP on the unconstrained space: in general, the MAP of
  the transform is not the transform of the MAP.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  """Create an inference algorithm.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized using a
      `PointMass` random variable that is defined internally with
      constrained support, has unconstrained free parameters, and is
      initialized using standard normal draws. If dictionary, each
      value in the dictionary must be a `PointMass` random variable
      with the same support as the key.
  """
  """Build loss function. Its automatic differentiation
  is the gradient of

  $- \log p(x,z).$
  """
  with Trace() as posterior_trace:
    call_function_up_to_args(variational, *args, **kwargs)
  intercept = make_intercept(
      posterior_trace, align_data, align_latent, args, kwargs)
  with Trace(intercept=intercept) as model_trace:
    call_function_up_to_args(model, *args, **kwargs)

  p_log_prob = 0.0
  for name, node in six.iteritems(model_trace):
    if align_latent(name) is not None or align_data(name) is not None:
      rv = node.value
      scale_factor = scale(name)
      p_log_prob += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if collections is not None:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=collections)
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=collections)

  loss = -p_log_prob + reg_penalty
  return loss
