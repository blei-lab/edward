from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_control_variate_scalar(f, h):
  """Returns scalar used by control variates method for variance reduction in
  MCMC methods.

  If we have a statistic :math:`m` unbiased estimator of :math:`\mu` and
  and another statistic :math:`t` which is an unbiased estimator of
  :math:`\tau` then :math:`m^* = m + c(t - \tau)` is also an unbiased
  estimator of :math:`\mu' for any coefficient :math:`c`.

  This function calculates the optimal coefficient
  .. math::

    \c^* = \frac{Cov(m,t)}{Var(t)}

  for minimizing the variance of :math:`m^*`.
  """
  f_mu = tf.reduce_mean(f)
  h_mu = tf.reduce_mean(h)

  n = f.get_shape()[0].value

  cov_fh = tf.reduce_sum((f - f_mu) * (h - h_mu)) / (n - 1)
  var_h = tf.reduce_sum(tf.square(h - h_mu)) / (n - 1)

  a = cov_fh / var_h

  return a
