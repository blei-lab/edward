from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


class GammaRejectionSampler:

  # As implemented in https://github.com/blei-lab/ars-reparameterization/blob/master/gamma/demo.ipynb

  def __init__(self, density):
    self.alpha = density.parameters['concentration']
    self.beta = density.parameters['rate']

  def h(self, epsilon):
    a = self.alpha - (1. / 3)
    b = tf.sqrt(9 * self.alpha - 3)
    c = 1 + (epsilon / b)
    d = a * c**3
    return d / self.beta

  def h_inverse(self, z):
    a = self.alpha - (1. / 3)
    b = tf.sqrt(9 * self.alpha - 3)
    c = self.beta * z / a
    d = c**(1 / 3)
    return b * (d - 1)

  @staticmethod
  def log_prob_s(epsilon):
    return -0.5 * (tf.log(2 * math.pi) + epsilon**2)
