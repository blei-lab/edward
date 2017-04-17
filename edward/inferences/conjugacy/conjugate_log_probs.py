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
def beta_log_prob(self, val):
  a = self.parameters['a']
  b = self.parameters['b']
  result = (a - 1.0) * tf.log(val)
  result += (b - 1.0) * tf.log(tf.constant(1.0) - val)
  result += -tf.lgamma(a) - tf.lgamma(b) + tf.lgamma(a + b)
  return result
rvs.Beta.conjugate_log_prob = beta_log_prob


@_val_wrapper
def dirichlet_log_prob(self, val):
  alpha = self.parameters['alpha']
  result = tf.reduce_sum((alpha - 1.0) * tf.log(val), -1)
  result += tf.reduce_sum(-tf.lgamma(alpha), -1)
  result += tf.lgamma(tf.reduce_sum(alpha, -1))
  return result
rvs.Dirichlet.conjugate_log_prob = dirichlet_log_prob


@_val_wrapper
def bernoulli_log_prob(self, val):
  p = self.parameters['p']
  f_val = tf.cast(val, tf.float32)
  return (f_val * tf.log(p) +
          (1. - f_val) * tf.log(1. - p))
rvs.Bernoulli.conjugate_log_prob = bernoulli_log_prob


@_val_wrapper
def categorical_log_prob(self, val):
  p = self.parameters['p']
  one_hot = tf.one_hot(val, p.get_shape()[-1], dtype=tf.float32)
  return tf.reduce_sum(tf.log(p) * one_hot, -1)
rvs.Categorical.conjugate_log_prob = categorical_log_prob


@_val_wrapper
def gamma_log_prob(self, val):
  alpha = self.parameters['alpha']
  beta = self.parameters['beta']
  result = (alpha - 1.0) * tf.log(val)
  result -= beta * val
  result += -tf.lgamma(alpha) + alpha * tf.log(beta)
  return result
rvs.Gamma.conjugate_log_prob = gamma_log_prob


@_val_wrapper
def poisson_log_prob(self, val):
  lam = self.parameters['lam']
  f_val = tf.cast(val, tf.float32)
  result = f_val * tf.log(lam)
  result += -lam - tf.lgamma(f_val + 1)
  return result
rvs.Poisson.conjugate_log_prob = poisson_log_prob


@_val_wrapper
def mvn_diag_log_prob(self, val):
  mu = self.parameters['mu']
  sigma = self.parameters['diag_stdev']
  prec = tf.reciprocal(tf.square(sigma))
  result = prec * (-0.5 * tf.square(val) - 0.5 * tf.square(mu) +
                   val * mu)
  result -= tf.log(sigma) + 0.5 * tf.log(2 * np.pi)
  return result
rvs.MultivariateNormalDiag.conjugate_log_prob = mvn_diag_log_prob


@_val_wrapper
def normal_log_prob(self, val):
  mu = self.parameters['mu']
  sigma = self.parameters['sigma']
  prec = tf.reciprocal(tf.square(sigma))
  result = prec * (-0.5 * tf.square(val) - 0.5 * tf.square(mu) +
                   val * mu)
  result -= tf.log(sigma) + 0.5 * tf.log(2 * np.pi)
  return result
rvs.Normal.conjugate_log_prob = normal_log_prob


@_val_wrapper
def inverse_gamma_log_prob(self, val):
  alpha = self.parameters['alpha']
  beta = self.parameters['beta']
  result = -(alpha + 1) * tf.log(val)
  result -= beta * tf.reciprocal(val)
  result += -tf.lgamma(alpha) + alpha * tf.log(beta)
  return result
rvs.InverseGamma.conjugate_log_prob = inverse_gamma_log_prob
