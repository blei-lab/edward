from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


class GammaRejectionSampler:

  # As implemented in https://github.com/blei-lab/ars-reparameterization/blob/master/gamma/demo.ipynb

  @staticmethod
  def h(epsilon, alpha, beta):
    a = alpha - (1. / 3)
    b = tf.sqrt(9 * alpha - 3)
    c = 1 + (epsilon / b)
    d = a * c**3
    return d / beta

  @staticmethod
  def h_inverse(z, alpha, beta):
    a = alpha - (1. / 3)
    b = tf.sqrt(9 * alpha - 3)
    c = beta * z / a
    d = c**(1 / 3)
    return b * (d - 1)

  @staticmethod
  def log_prob_s(epsilon):
    return -0.5 * (tf.log(2 * math.pi) + epsilon**2)
