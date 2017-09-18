from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf
from collections import namedtuple
from edward.models import (
  Beta,
  Dirichlet,
  Gamma,
  MultivariateNormalDiag,
  Normal,
  PointMass,
  TransformedDistribution,
)
from tensorflow.contrib.distributions import bijectors

class test_transform_class(tf.test.TestCase):

  def test_args(self):
    with self.test_session():
      x = Normal(0.0, 1.0)
      y = ed.transform(x, bijectors.Softplus())
      y.eval()

  def test_kwargs(self):
    with self.test_session():
      x = Normal(0.0, 1.0)
      y = ed.transform(x, bijector=bijectors.Softplus())
      y.eval()

  def test_01(self):
    with self.test_session():
      x = Beta(1.0, 1.0)
      y = ed.transform(x)
      self.assertIsInstance(y, TransformedDistribution)
      y.eval()

  def test_nonnegative(self):
    with self.test_session():
      x = Gamma(1.0, 1.0)
      y = ed.transform(x)
      self.assertIsInstance(y, TransformedDistribution)
      y.eval()

  def test_simplex(self):
    with self.test_session():
      x = Dirichlet(tf.zeros(5))
      y = ed.transform(x)
      self.assertIsInstance(y, TransformedDistribution)
      y.eval()

  def test_real(self):
    with self.test_session():
      x = Normal(0.0, 1.0)
      y = ed.transform(x)
      self.assertIsInstance(y, Normal)
      y.eval()

  def test_multivariate_real(self):
    with self.test_session():
      x = MultivariateNormalDiag(tf.zeros(2), tf.ones(2))
      y = ed.transform(x)
      y.eval()

  def test_no_support(self):
    with self.test_session():
      x = PointMass(1.0)
      with self.assertRaises(ValueError):
        y = ed.transform(x)

  def test_unhandled_support(self):
    with self.test_session():
      FakeRV = namedtuple('FakeRV', ['support'])
      x = FakeRV(support='rational')
      with self.assertRaises(NotImplementedError):
        y = ed.transform(x)

if __name__ == '__main__':
  tf.test.main()
