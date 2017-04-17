from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import random_variables as rvs


def _val_wrapper(f):
  def wrapped(self, val=None):
    if val is None:
      return f(self, self)
    else:
      return f(self, val)
  return wrapped


@_val_wrapper
def bernoulli_log_prob(self, val):
  p = self.parameters['p']
  f_val = tf.cast(val, tf.float32)
  return f_val * tf.log(p) + (1.0 - f_val) * tf.log(1.0 - p)


@_val_wrapper
def beta_log_prob(self, val):
  a = self.parameters['a']
  b = self.parameters['b']
  result = (a - 1.0) * tf.log(val)
  result += (b - 1.0) * tf.log(tf.constant(1.0) - val)
  result += -tf.lgamma(a) - tf.lgamma(b) + tf.lgamma(a + b)
  return result


@_val_wrapper
def binomial_log_prob(self, val):
  n = self.parameters['n']
  p = self.parameters['p']
  f_n = tf.cast(n, tf.float32)
  f_val = tf.cast(val, tf.float32)
  result = f_val * tf.log(p) + (f_n - f_val) * tf.log(1.0 - p)
  result += tf.lgamma(f_n + 1) - tf.lgamma(f_val + 1) - \
      tf.lgamma(f_n - f_val + 1)
  return result


@_val_wrapper
def categorical_log_prob(self, val):
  p = self.parameters['p']
  one_hot = tf.one_hot(val, p.get_shape()[-1], dtype=tf.float32)
  return tf.reduce_sum(tf.log(p) * one_hot, -1)


@_val_wrapper
def chi2_log_prob(self, val):
  df = self.parameters['df']
  eta = 0.5 * df - 1
  result = tf.reduce_sum(eta * tf.log(val), -1)
  result += tf.exp(-0.5 * val)
  result -= tf.lgamma(eta + 1) + (eta + 1) * tf.log(2.0)
  return result


@_val_wrapper
def dirichlet_log_prob(self, val):
  alpha = self.parameters['alpha']
  result = tf.reduce_sum((alpha - 1.0) * tf.log(val), -1)
  result += tf.reduce_sum(-tf.lgamma(alpha), -1)
  result += tf.lgamma(tf.reduce_sum(alpha, -1))
  return result


@_val_wrapper
def exponential_log_prob(self, val):
  lam = self.parameters['lam']
  result = tf.log(lam) - lam * val
  return result


@_val_wrapper
def gamma_log_prob(self, val):
  alpha = self.parameters['alpha']
  beta = self.parameters['beta']
  result = (alpha - 1.0) * tf.log(val)
  result -= beta * val
  result += -tf.lgamma(alpha) + alpha * tf.log(beta)
  return result


@_val_wrapper
def inverse_gamma_log_prob(self, val):
  alpha = self.parameters['alpha']
  beta = self.parameters['beta']
  result = -(alpha + 1) * tf.log(val)
  result -= beta * tf.reciprocal(val)
  result += -tf.lgamma(alpha) + alpha * tf.log(beta)
  return result


@_val_wrapper
def laplace_log_prob(self, val):
  loc = self.parameters['loc']
  scale = self.parameters['scale']
  f_val = tf.cast(val, tf.float32)
  result = -tf.log(2.0 * scale) - tf.abs(f_val - loc) / scale
  return result


@_val_wrapper
def multinomial_log_prob(self, val):
  n = self.parameters['n']
  p = self.parameters['p']
  f_n = tf.cast(n, tf.float32)
  f_val = tf.cast(val, tf.float32)
  result = tf.reduce_sum(tf.log(p) * f_val, -1)
  result += tf.lgamma(f_n + 1) - tf.reduce_sum(tf.lgamma(f_val + 1), -1)
  return result


@_val_wrapper
def mvn_diag_log_prob(self, val):
  mu = self.parameters['mu']
  sigma = self.parameters['diag_stdev']
  prec = tf.reciprocal(tf.square(sigma))
  result = prec * (-0.5 * tf.square(val) - 0.5 * tf.square(mu) +
                   val * mu)
  result -= tf.log(sigma) + 0.5 * tf.log(2 * np.pi)
  return result


@_val_wrapper
def normal_log_prob(self, val):
  mu = self.parameters['mu']
  sigma = self.parameters['sigma']
  prec = tf.reciprocal(tf.square(sigma))
  result = prec * (-0.5 * tf.square(val) - 0.5 * tf.square(mu) +
                   val * mu)
  result -= tf.log(sigma) + 0.5 * tf.log(2 * np.pi)
  return result


@_val_wrapper
def poisson_log_prob(self, val):
  lam = self.parameters['lam']
  f_val = tf.cast(val, tf.float32)
  result = f_val * tf.log(lam)
  result += -lam - tf.lgamma(f_val + 1)
  return result


rvs.Bernoulli.conjugate_log_prob = bernoulli_log_prob
rvs.Beta.conjugate_log_prob = beta_log_prob
rvs.Binomial.conjugate_log_prob = binomial_log_prob
rvs.Categorical.conjugate_log_prob = categorical_log_prob
rvs.Chi2.conjugate_log_prob = chi2_log_prob
rvs.Dirichlet.conjugate_log_prob = dirichlet_log_prob
rvs.Exponential.conjugate_log_prob = exponential_log_prob
rvs.Gamma.conjugate_log_prob = gamma_log_prob
rvs.InverseGamma.conjugate_log_prob = inverse_gamma_log_prob
rvs.Laplace.conjugate_log_prob = laplace_log_prob
rvs.Multinomial.conjugate_log_prob = multinomial_log_prob
rvs.MultivariateNormalDiag.conjugate_log_prob = mvn_diag_log_prob
rvs.Normal.conjugate_log_prob = normal_log_prob
rvs.Poisson.conjugate_log_prob = poisson_log_prob
