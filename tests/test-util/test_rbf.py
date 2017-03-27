from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import rbf


class test_rbf_class(tf.test.TestCase):

  def test_x(self):
    with self.test_session():
      X = tf.constant([[0.0], [0.0]])
      X2 = tf.constant([[0.0], [0.0]])
      self.assertAllClose(rbf(X).eval(),
                          [[1.0, 1.0], [1.0, 1.0]])
      self.assertAllClose(rbf(X, X2).eval(),
                          [[1.0, 1.0], [1.0, 1.0]])

  def test_x2(self):
    with self.test_session():
      X = tf.constant([[10.0], [2.0]])
      X2 = tf.constant([[2.0], [10.0]])
      self.assertAllClose(rbf(X, X2).eval(),
                          [[1.266417e-14, 1.0], [1.0, 1.266417e-14]])
      self.assertAllClose(rbf(X2, X).eval(),
                          [[1.266417e-14, 1.0], [1.0, 1.266417e-14]])

      X = tf.constant([[2.0, 2.5], [4.1, 5.0]])
      X2 = tf.constant([[1.5, 2.0], [3.1, 4.2]])
      self.assertAllClose(rbf(X, X2).eval(),
                          [[0.778800, 0.128734],
                           [0.000378, 0.440431]], atol=1e-5, rtol=1e-5)

  def test_lengthscale(self):
    """checked calculations by hand, e.g.,
    np.exp(-((2.0 - 1.5)**2 / (2.0**2) + (2.5 - 2.0)**2 / (1.5**2)) / 2)
    np.exp(-((2.0 - 3.1)**2 / (2.0**2) + (2.5 - 4.2)**2 / (1.5**2)) / 2)
    np.exp(-((4.1 - 1.5)**2 / (2.0**2) + (5.0 - 2.0)**2 / (1.5**2)) / 2)
    np.exp(-((4.1 - 3.1)**2 / (2.0**2) + (5.0 - 4.2)**2 / (1.5**2)) / 2)
    """
    with self.test_session():
      X = tf.constant([[2.0, 2.5], [4.1, 5.0]])
      X2 = tf.constant([[1.5, 2.0], [3.1, 4.2]])
      lengthscale1 = tf.constant(2.0)
      lengthscale2 = tf.constant([2.0, 2.0])
      lengthscale3 = tf.constant([2.0, 1.5])
      self.assertAllClose(rbf(X, X2, lengthscale1).eval(),
                          [[0.939413, 0.598996],
                           [0.139456, 0.814647]], atol=1e-5, rtol=1e-5)
      self.assertAllClose(rbf(X, X2, lengthscale2).eval(),
                          [[0.939413, 0.598996],
                           [0.139456, 0.814647]], atol=1e-5, rtol=1e-5)
      self.assertAllClose(rbf(X, X2, lengthscale3).eval(),
                          [[0.916855, 0.452271],
                           [0.058134, 0.765502]], atol=1e-5, rtol=1e-5)

  def test_variance(self):
    with self.test_session():
      X = tf.constant([[2.0, 2.5], [4.1, 5.0]])
      X2 = tf.constant([[1.5, 2.0], [3.1, 4.2]])
      variance = tf.constant(1.4)
      self.assertAllClose(rbf(X, X2, variance=variance).eval(),
                          [[1.090321, 0.180228],
                           [0.000529, 0.616604]], atol=1e-5, rtol=1e-5)

  def test_all(self):
    with self.test_session():
      X = tf.constant([[2.0, 2.5], [4.1, 5.0]])
      X2 = tf.constant([[1.5, 2.0], [3.1, 4.2]])
      lengthscale = tf.constant([2.0, 1.5])
      variance = tf.constant(1.4)
      self.assertAllClose(rbf(X, X2, lengthscale, variance).eval(),
                          [[1.283597, 0.633180],
                           [0.081387, 1.071704]], atol=1e-5, rtol=1e-5)

  def test_raises(self):
    with self.test_session():
      X1 = tf.constant([[0.0]])
      X2 = tf.constant([[0.0]])
      lengthscale = tf.constant(-5.0)
      variance = tf.constant(-1.0)
      with self.assertRaisesOpError('Condition'):
        rbf(X1, X2, variance=variance).eval()
        rbf(X1, X2, lengthscale).eval()
        rbf(X1, X2, lengthscale, variance).eval()

if __name__ == '__main__':
  tf.test.main()
