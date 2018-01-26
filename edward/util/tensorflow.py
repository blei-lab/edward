from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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
