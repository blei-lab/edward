from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def dot(x, y):
  """Compute dot product between a 2-D tensor and a 1-D tensor.

  If x is a `[M x N]` matrix, then y is a `M`-vector.

  If x is a `M`-vector, then y is a `[M x N]` matrix.

  Args:
    x: tf.Tensor.
      A 1-D or 2-D tensor (see above).
    y: tf.Tensor.
      A 1-D or 2-D tensor (see above).

  Returns:
    tf.Tensor.
    A 1-D tensor of length `N`.

  Raises:
    InvalidArgumentError.
    If the inputs have Inf or NaN values.
  """
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)
  dependencies = [tf.verify_tensor_all_finite(x, msg=''),
                  tf.verify_tensor_all_finite(y, msg='')]
  x = control_flow_ops.with_dependencies(dependencies, x)
  y = control_flow_ops.with_dependencies(dependencies, y)

  if len(x.shape) == 1:
    vec = x
    mat = y
    return tf.reshape(tf.matmul(tf.expand_dims(vec, 0), mat), [-1])
  else:
    mat = x
    vec = y
    return tf.reshape(tf.matmul(mat, tf.expand_dims(vec, 1)), [-1])


def get_control_variate_coef(f, h):
  """Returns scalar used by control variates method for variance reduction in
  Monte Carlo methods.

  If we have a statistic $m$ as an unbiased estimator of $\mu$ and
  and another statistic $t$ which is an unbiased estimator of
  $\\tau$ then $m^* = m + c(t - \\tau)$ is also an unbiased
  estimator of $\mu$ for any coefficient $c$.

  This function calculates the optimal coefficient

  $c^* = \\frac{\\text{Cov}(m,t)}{\\text{Var}(t)}$

  for minimizing the variance of $m^*$.

  Args:
    f: tf.Tensor.
      A 1-D tensor.
    h: tf.Tensor.
      A 1-D tensor.

  Returns:
    tf.Tensor.
    A 0 rank tensor
  """
  f_mu = tf.reduce_mean(f)
  h_mu = tf.reduce_mean(h)

  n = f.shape[0].value

  cov_fh = tf.reduce_sum((f - f_mu) * (h - h_mu)) / (n - 1)
  var_h = tf.reduce_sum(tf.square(h - h_mu)) / (n - 1)

  a = cov_fh / var_h

  return a
