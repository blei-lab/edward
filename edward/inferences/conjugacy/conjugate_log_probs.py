from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models.random_variable import RandomVariable
from edward.models import random_variables as rvs

def _canonical_value(x):
  return x
#   if isinstance(x, RandomVariable):
#     return x.value()
#   else:
#     return x


def beta_log_prob(self):
  val = self
  a = _canonical_value(self.parameters['a'])
  b = _canonical_value(self.parameters['b'])
  result = ((a) - 1) * tf.log(val)
  result += ((b) - 1) * tf.log(1. - (val))
  result += -tf.lgamma(a) - tf.lgamma(b) + tf.lgamma(a + b)
  return result
rvs.Beta.conjugate_log_prob = beta_log_prob


def bernoulli_log_prob(self):
  val = self
  p = _canonical_value(self.parameters['p'])
  f_val = tf.cast(val, np.float32)
  return ((f_val) * tf.log(p) +
          (1. - (f_val)) * tf.log(1. - p))
rvs.Bernoulli.conjugate_log_prob = bernoulli_log_prob


def gamma_log_prob(self):
  val = self
  alpha = _canonical_value(self.parameters['alpha'])
  beta = _canonical_value(self.parameters['beta'])
  result = ((alpha) - 1) * tf.log(val)
  result -= (beta) * (val)
  result += -tf.lgamma(alpha) + (alpha) * tf.log(beta)
  return result
rvs.Gamma.conjugate_log_prob = gamma_log_prob


def poisson_log_prob(self):
  val = self
  lam = _canonical_value(self.parameters['lam'])
  f_val = tf.cast(val, np.float32)
  result = (f_val) * tf.log(lam)
  result += -lam - tf.lgamma(f_val+1)
  return result
rvs.Poisson.conjugate_log_prob = poisson_log_prob


def normal_log_prob(self):
  val = self
  mu = _canonical_value(self.parameters['mu'])
  sigma = _canonical_value(self.parameters['sigma'])
  prec = tf.reciprocal(tf.square(sigma))
  result = prec * (-0.5 * tf.square(val) - 0.5 * tf.square(mu)
                   + (val) * (mu))
  result -= tf.log(sigma) + 0.5 * np.log(2*np.pi)
  return result
rvs.Normal.conjugate_log_prob = normal_log_prob


def inverse_gamma_log_prob(self):
  val = self
  alpha = _canonical_value(self.parameters['alpha'])
  beta = _canonical_value(self.parameters['beta'])
  result = -((alpha) + 1) * tf.log(val)
  result -= (beta) * tf.reciprocal(val)
  result += -tf.lgamma(alpha) + (alpha) * tf.log(beta)
  return result
rvs.InverseGamma.conjugate_log_prob = inverse_gamma_log_prob
