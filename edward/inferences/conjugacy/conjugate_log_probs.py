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
  probs = self.parameters['probs']
  f_val = tf.cast(val, tf.float32)
  return f_val * tf.log(probs) + (1.0 - f_val) * tf.log(1.0 - probs)


@_val_wrapper
def beta_log_prob(self, val):
  conc0 = self.parameters['concentration0']
  conc1 = self.parameters['concentration1']
  result = (conc1 - 1.0) * tf.log(val)
  result += (conc0 - 1.0) * tf.log(1.0 - val)
  result += -tf.lgamma(conc1) - tf.lgamma(conc0) + tf.lgamma(conc1 + conc0)
  return result


@_val_wrapper
def binomial_log_prob(self, val):
  n = self.parameters['total_count']
  probs = self.parameters['probs']
  f_n = tf.cast(n, tf.float32)
  f_val = tf.cast(val, tf.float32)
  result = f_val * tf.log(probs) + (f_n - f_val) * tf.log(1.0 - probs)
  result += tf.lgamma(f_n + 1) - tf.lgamma(f_val + 1) - \
      tf.lgamma(f_n - f_val + 1)
  return result


@_val_wrapper
def categorical_log_prob(self, val):
  probs = self.parameters['probs']
  one_hot = tf.one_hot(val, probs.shape[-1], dtype=tf.float32)
  return tf.reduce_sum(tf.log(probs) * one_hot, -1)


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
  conc = self.parameters['concentration']
  result = tf.reduce_sum((conc - 1.0) * tf.log(val), -1)
  result += tf.reduce_sum(-tf.lgamma(conc), -1)
  result += tf.lgamma(tf.reduce_sum(conc, -1))
  return result


@_val_wrapper
def exponential_log_prob(self, val):
  rate = self.parameters['rate']
  result = tf.log(rate) - rate * val
  return result


@_val_wrapper
def gamma_log_prob(self, val):
  conc = self.parameters['concentration']
  rate = self.parameters['rate']
  result = (conc - 1.0) * tf.log(val)
  result -= rate * val
  result += -tf.lgamma(conc) + conc * tf.log(rate)
  return result


@_val_wrapper
def inverse_gamma_log_prob(self, val):
  conc = self.parameters['concentration']
  rate = self.parameters['rate']
  result = -(conc + 1) * tf.log(val)
  result -= rate * tf.reciprocal(val)
  result += -tf.lgamma(conc) + conc * tf.log(rate)
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
  n = self.parameters['total_count']
  probs = self.parameters['probs']
  f_n = tf.cast(n, tf.float32)
  f_val = tf.cast(val, tf.float32)
  result = tf.reduce_sum(tf.log(probs) * f_val, -1)
  result += tf.lgamma(f_n + 1) - tf.reduce_sum(tf.lgamma(f_val + 1), -1)
  return result


@_val_wrapper
def mvn_diag_log_prob(self, val):
  loc = self.parameters['loc']
  scale_diag = self.parameters['scale_diag']
  prec = tf.reciprocal(tf.square(scale_diag))
  result = prec * (-0.5 * tf.square(val) - 0.5 * tf.square(loc) +
                   val * loc)
  result -= tf.log(scale_diag) + 0.5 * tf.log(2 * np.pi)
  return result


@_val_wrapper
def normal_log_prob(self, val):
  loc = self.parameters['loc']
  scale = self.parameters['scale']
  prec = tf.reciprocal(tf.square(scale))
  result = prec * (-0.5 * tf.square(val) - 0.5 * tf.square(loc) +
                   val * loc)
  result -= tf.log(scale) + 0.5 * tf.log(2 * np.pi)
  return result


@_val_wrapper
def poisson_log_prob(self, val):
  rate = self.parameters['rate']
  f_val = tf.cast(val, tf.float32)
  result = f_val * tf.log(rate)
  result += -rate - tf.lgamma(f_val + 1)
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
