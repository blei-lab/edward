from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import call_function_up_to_args
from edward.inferences import docstrings as doc
from edward.inferences.map import map
from edward.models.core import Trace
from edward.models.queries import get_variables

try:
  from edward.models import \
      MultivariateNormalDiag, MultivariateNormalTriL, Normal
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


def laplace(model, variational, align_latent, align_data,
            scale=lambda name: 1.0, auto_transform=True,
            collections=None, *args, **kwargs):
  """Laplace approximation [@laplace1986memoir].

  It approximates the posterior distribution using a multivariate
  normal distribution centered at the mode of the posterior.

  We implement this by running `MAP` to find the posterior mode.
  This forms the mean of the normal approximation. We then compute the
  inverse Hessian at the mode of the posterior. This forms the
  covariance of the normal approximation.

  #### Notes

  If `MultivariateNormalDiag` or `Normal` random variables are
  specified as approximations, then the Laplace approximation will
  only produce the diagonal. This does not capture correlation among
  the variables but it does not require a potentially expensive
  matrix inversion.

  Random variables with both scalar batch and event shape are not
  supported as `tf.hessians` is currently not applicable to scalars.

  Note that `Laplace` finds the location parameter of the normal
  approximation using `MAP`, which is performed on the latent
  variable's original (constrained) support. The scale parameter
  is calculated by evaluating the Hessian of $-\log p(x, z)$ in the
  constrained space and under the mode. This implies the Laplace
  approximation always has real support even if the target
  distribution has constrained support.

  #### Examples

  ```python
  X = tf.placeholder(tf.float32, [N, D])
  w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
  y = Normal(loc=ed.dot(X, w), scale=tf.ones(N))

  qw = MultivariateNormalTriL(
      loc=tf.Variable(tf.random_normal([D])),
      scale_tril=tf.Variable(tf.random_normal([D, D])))

  inference = ed.Laplace({w: qw}, data={X: X_train, y: y_train})
  ```
  """
  """Create an inference algorithm.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If list,
      each random variable will be implictly optimized using a
      `MultivariateNormalTriL` random variable that is defined
      internally with unconstrained support and is initialized using
      standard normal draws. If dictionary, each random
      variable must be a `MultivariateNormalDiag`,
      `MultivariateNormalTriL`, or `Normal` random variable.
  """
  variational_pointmass = _make_variational_pointmass(
      variational, *args, **kwargs)
  loss = map(model, variational, align_latent, align_data,
             scale, auto_transform, collections, *args, **kwargs)
  finalize_op = _finalize(loss, variational)
  return loss, finalize_op


def _finalize(loss, variational):
  """Function to call after convergence.

  Computes the Hessian at the mode.
  """
  with Trace() as trace:
    call_function_up_to_args(variational, *args, **kwargs)
  hessians = tf.hessians(
      loss, [node.value.loc for node in six.itervalues(trace)])
  finalize_ops = []
  for qz, hessian in zip(six.itervalues(trace), hessians):
    if isinstance(qz, (MultivariateNormalDiag, Normal)):
      scale_var = get_variables(qz.variance())[0]
      scale = 1.0 / tf.diag_part(hessian)
    else:  # qz is MultivariateNormalTriL
      scale_var = get_variables(qz.covariance())[0]
      scale = tf.matrix_inverse(tf.cholesky(hessian))

    finalize_ops.append(scale_var.assign(scale))
  return tf.group(*finalize_ops)


def _make_variational_pointmass(variational, *args, **kwargs):
  """Take a variational program and build a new one that replaces all
  random variables with point masses.

  We assume all latent variables are traceable in one execution.
  """
  with Trace() as trace:
    call_function_up_to_args(variational, *args, **kwargs)

  def variational_pointmass(*args, **kwargs):
    for name, node in six.iteritems(trace):
      qz = node.value
      qz_pointmass = PointMass(params=qz.loc,
                               name=qz.name + "_pointmass",
                               value=qz.loc)
  return variational_pointmass
