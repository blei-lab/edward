from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from collections import namedtuple
from edward.models import (
    Beta, Dirichlet, DirichletProcess, Gamma, MultivariateNormalDiag,
    Normal, Poisson, TransformedDistribution)
from tensorflow.contrib.distributions import bijectors


class test_transform_class(tf.test.TestCase):

  def assertSamplePosNeg(self, sample):
    num_pos = np.sum((sample > 0.0), axis=0, keepdims=True)
    num_neg = np.sum((sample < 0.0), axis=0, keepdims=True)
    self.assertTrue((num_pos > 0).all())
    self.assertTrue((num_neg > 0).all())

  def test_args(self):
    with self.test_session():
      x = Normal(-100.0, 1.0)
      y = ed.transform(x, bijectors.Softplus())
      sample = y.sample(10).eval()
      self.assertTrue((sample >= 0.0).all())

  def test_kwargs(self):
    with self.test_session():
      x = Normal(-100.0, 1.0)
      y = ed.transform(x, bijector=bijectors.Softplus())
      sample = y.sample(10).eval()
      self.assertTrue((sample >= 0.0).all())

  def test_01(self):
    with self.test_session():
      x = Beta(1.0, 1.0)
      y = ed.transform(x)
      self.assertIsInstance(y, TransformedDistribution)
      sample = y.sample(10, seed=1).eval()
      self.assertSamplePosNeg(sample)

  def test_nonnegative(self):
    with self.test_session():
      x = Gamma(1.0, 1.0)
      y = ed.transform(x)
      self.assertIsInstance(y, TransformedDistribution)
      sample = y.sample(10, seed=1).eval()
      self.assertSamplePosNeg(sample)

  def test_simplex(self):
    with self.test_session():
      x = Dirichlet([1.1, 1.2, 1.3, 1.4])
      y = ed.transform(x)
      self.assertIsInstance(y, TransformedDistribution)
      sample = y.sample(10, seed=1).eval()
      self.assertSamplePosNeg(sample)

  def test_real(self):
    with self.test_session():
      x = Normal(0.0, 1.0)
      y = ed.transform(x)
      self.assertIsInstance(y, Normal)
      sample = y.sample(10, seed=1).eval()
      self.assertSamplePosNeg(sample)

  def test_multivariate_real(self):
    with self.test_session():
      x = MultivariateNormalDiag(tf.zeros(2), tf.ones(2))
      y = ed.transform(x)
      sample = y.sample(10, seed=1).eval()
      self.assertSamplePosNeg(sample)

  def test_no_support(self):
    with self.test_session():
      x = DirichletProcess(1.0, Normal(0.0, 1.0))
      with self.assertRaises(AttributeError):
        y = ed.transform(x)

  def test_unhandled_support(self):
    with self.test_session():
      FakeRV = namedtuple('FakeRV', ['support'])
      x = FakeRV(support='rational')
      with self.assertRaises(ValueError):
        y = ed.transform(x)

if __name__ == '__main__':
  tf.test.main()
