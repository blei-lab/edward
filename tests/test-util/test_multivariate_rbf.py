from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import multivariate_rbf


class test_multivariate_rbf_class_class(tf.test.TestCase):

  def test_multivariate_rbf_0d(self):
    with self.test_session():
      x = tf.constant(0.0)
      self.assertAllClose(multivariate_rbf(x).eval(),
                          1.0)
      x = tf.constant(10.0)
      y = tf.constant(2.0)
      self.assertAllClose(multivariate_rbf(x, y=y).eval(),
                          1.26e-14)
      x = tf.constant(0.0)
      y = tf.constant(1.0)
      sigma = tf.constant(10.0)
      self.assertAllClose(multivariate_rbf(x, y=y, sigma=sigma).eval(),
                          60.6530685)
      x = tf.constant(0.0)
      y = tf.constant(1.0)
      sigma = tf.constant(10.0)
      l = tf.constant(5.0)
      self.assertAllClose(multivariate_rbf(x, y=y, sigma=sigma, l=l).eval(),
                          98.01986694)

  def test_multivariate_rbf_1d(self):
    with self.test_session():
      x = tf.constant([0.0])
      self.assertAllClose(multivariate_rbf(x).eval(),
                          1.0)
      x = tf.constant([10.0])
      y = tf.constant([2.0])
      self.assertAllClose(multivariate_rbf(x, y=y).eval(),
                          1.26e-14)
      x = tf.constant([0.0])
      y = tf.constant([1.0])
      sigma = tf.constant(10.0)
      self.assertAllClose(multivariate_rbf(x, y=y, sigma=sigma).eval(),
                          60.6530685)
      x = tf.constant([0.0])
      y = tf.constant([1.0])
      sigma = tf.constant(10.0)
      l = tf.constant(5.0)
      self.assertAllClose(multivariate_rbf(x, y=y, sigma=sigma, l=l).eval(),
                          98.01986694)
      x = tf.constant([0.0, 1.0])
      self.assertAllClose(multivariate_rbf(x).eval(),
                          0.606530666)
      x = tf.constant([10.0, 3.0])
      y = tf.constant([2.0, 3.0])
      self.assertAllClose(multivariate_rbf(x, y=y).eval(),
                          1.26e-14)
      x = tf.constant([0.0, 1.0])
      y = tf.constant([1.0, 2.0])
      sigma = tf.constant(10.0)
      self.assertAllClose(multivariate_rbf(x, y=y, sigma=sigma).eval(),
                          36.7879447)
      x = tf.constant([0.0, -23.0])
      y = tf.constant([1.0, -93.0])
      sigma = tf.constant(10.0)
      l = tf.constant(50.0)
      self.assertAllClose(multivariate_rbf(x, y=y, sigma=sigma, l=l).eval(),
                          37.52360534)

  def test_multivariate_rbf_2d(self):
    with self.test_session():
      x = tf.constant([[0.0], [0.0]])
      self.assertAllClose(multivariate_rbf(x).eval(),
                          1.0)
      x = tf.constant([[10.0], [2.0]])
      y = tf.constant([[2.0], [10.0]])
      self.assertAllClose(multivariate_rbf(x, y=y).eval(),
                          1.26e-14)
      x = tf.constant([[0.0], [10.0]])
      y = tf.constant([[1.0], [1.0]])
      sigma = tf.constant(10.0)
      self.assertAllClose(multivariate_rbf(x, y=y, sigma=sigma).eval(),
                          1.562882e-16)
      x = tf.constant([[0.0], [10.0]])
      y = tf.constant([[1.0], [1.0]])
      sigma = tf.constant(10.0)
      l = tf.constant(5.0)
      self.assertAllClose(multivariate_rbf(x, y=y, sigma=sigma, l=l).eval(),
                          19.39800453)
      x = tf.constant([[10.0, 3.0], [10.0, 3.0]])
      self.assertAllClose(multivariate_rbf(x).eval(),
                          0.0)
      x = tf.constant([[10.0, 3.0], [10.0, 3.0]])
      y = tf.constant([[2.0, 3.0], [2.0, 3.0]])
      self.assertAllClose(multivariate_rbf(x, y=y).eval(),
                          1.26e-14)
      x = tf.constant([[0.0, 1.0], [10.0, -3.0]])
      y = tf.constant([[1.0, 2.0], [1.0, 2.0]])
      sigma = tf.constant(10.0)
      self.assertAllClose(multivariate_rbf(x, y=y, sigma=sigma).eval(),
                          3.532628e-22)
      x = tf.constant([[10.0, 3.0], [10.0, 3.0]])
      y = tf.constant([[2.0, 3.0], [2.0, 3.0]])
      sigma = tf.constant(10.0)
      l = tf.constant(5.0)
      self.assertAllClose(multivariate_rbf(x, y=y, sigma=sigma, l=l).eval(),
                          7.730474472)

  def test_contraint_raises(self):
    with self.test_session():
      x = tf.constant(0.0)
      y = tf.constant(1.0)
      sigma = tf.constant(-1.0)
      l = tf.constant(-5.0)
      with self.assertRaisesOpError('Condition'):
        multivariate_rbf(x, y=y, sigma=sigma).eval()
        multivariate_rbf(x, y=y, l=l).eval()
        multivariate_rbf(x, y=y, sigma=sigma, l=l).eval()
      x = np.inf * tf.constant(1.0)
      y = tf.constant(1.0)
      sigma = tf.constant(1.0)
      l = tf.constant(5.0)
      with self.assertRaisesOpError('Inf'):
        multivariate_rbf(x).eval()
        multivariate_rbf(x, y=y).eval()
        multivariate_rbf(x, y=y, sigma=sigma).eval()
        multivariate_rbf(x, y=y, l=l).eval()
        multivariate_rbf(x, y=y, sigma=sigma, l=l).eval()
      x = tf.constant(0.0)
      y = np.nan * tf.constant(1.0)
      sigma = tf.constant(1.0)
      l = tf.constant(5.0)
      with self.assertRaisesOpError('NaN'):
        multivariate_rbf(x, y=y).eval()
        multivariate_rbf(x, y=y, sigma=sigma).eval()
        multivariate_rbf(x, y=y, l=l).eval()
        multivariate_rbf(x, y=y, sigma=sigma, l=l).eval()

if __name__ == '__main__':
  tf.test.main()
