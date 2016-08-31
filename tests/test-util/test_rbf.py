from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import rbf


class test_rbf_class(tf.test.TestCase):

  def test_rbf_0d(self):
    with self.test_session():
      x = tf.constant(0.0)
      self.assertAllClose(rbf(x).eval(),
                          1.0)
      x = tf.constant(10.0)
      y = tf.constant(2.0)
      self.assertAllClose(rbf(x, y=y).eval(),
                          1.26e-14)
      x = tf.constant(0.0)
      y = tf.constant(1.0)
      sigma = tf.constant(10.0)
      self.assertAllClose(rbf(x, y=y, sigma=sigma).eval(),
                          60.6530685)
      x = tf.constant(0.0)
      y = tf.constant(1.0)
      sigma = tf.constant(10.0)
      l = tf.constant(5.0)
      self.assertAllClose(rbf(x, y=y, sigma=sigma, l=l).eval(),
                          98.01986694)

  def test_rbf_1d(self):
    with self.test_session():
      x = tf.constant([0.0])
      self.assertAllClose(rbf(x).eval(),
                          [1.0])
      x = tf.constant([10.0])
      y = tf.constant([2.0])
      self.assertAllClose(rbf(x, y=y).eval(),
                          [1.26e-14])
      x = tf.constant([0.0])
      y = tf.constant([1.0])
      sigma = tf.constant(10.0)
      self.assertAllClose(rbf(x, y=y, sigma=sigma).eval(),
                          [60.6530685])
      x = tf.constant([0.0])
      y = tf.constant([1.0])
      sigma = tf.constant(10.0)
      l = tf.constant(5.0)
      self.assertAllClose(rbf(x, y=y, sigma=sigma, l=l).eval(),
                          [98.01986694])
      x = tf.constant([0.0, 1.0])
      self.assertAllClose(rbf(x).eval(),
                          [1.0, 0.606530666])
      x = tf.constant([10.0, 3.0])
      y = tf.constant([2.0, 3.0])
      self.assertAllClose(rbf(x, y=y).eval(),
                          [1.266417e-14, 1.0])
      x = tf.constant([0.0, 1.0])
      y = tf.constant([1.0, 2.0])
      sigma = tf.constant(10.0)
      self.assertAllClose(rbf(x, y=y, sigma=sigma).eval(),
                          [60.6530685, 60.6530685])
      x = tf.constant([0.0, -23.0])
      y = tf.constant([1.0, -93.0])
      sigma = tf.constant(10.0)
      l = tf.constant(50.0)
      self.assertAllClose(rbf(x, y=y, sigma=sigma, l=l).eval(),
                          [99.980003, 37.531109])

  def test_rbf_2d(self):
    with self.test_session():
      x = tf.constant([[0.0], [0.0]])
      self.assertAllClose(rbf(x).eval(),
                          [[1.0], [1.0]])
      x = tf.constant([[10.0], [2.0]])
      y = tf.constant([[2.0], [10.0]])
      self.assertAllClose(rbf(x, y=y).eval(),
                          [[1.266417e-14], [1.266417e-14]])
      x = tf.constant([[0.0], [10.0]])
      y = tf.constant([[1.0], [1.0]])
      sigma = tf.constant(10.0)
      self.assertAllClose(rbf(x, y=y, sigma=sigma).eval(),
                          [[6.065307e+01], [2.576757e-16]])
      x = tf.constant([[0.0], [10.0]])
      y = tf.constant([[1.0], [1.0]])
      sigma = tf.constant(10.0)
      l = tf.constant(5.0)
      self.assertAllClose(rbf(x, y=y, sigma=sigma, l=l).eval(),
                          [[98.019867], [19.789869]])
      x = tf.constant([[10.0, 3.0], [10.0, 3.0]])
      self.assertAllClose(rbf(x, y=y).eval(),
                          [[2.576757e-18, 1.353353e-01],
                           [2.576757e-18, 1.353353e-01]])
      x = tf.constant([[10.0, 3.0], [10.0, 3.0]])
      y = tf.constant([[2.0, 3.0], [2.0, 3.0]])
      self.assertAllClose(rbf(x, y=y).eval(),
                          [[1.266417e-14, 1.0],
                           [1.266417e-14, 1.0]])
      x = tf.constant([[0.0, 1.0], [10.0, -3.0]])
      y = tf.constant([[1.0, 2.0], [1.0, 2.0]])
      sigma = tf.constant(10.0)
      self.assertAllClose(rbf(x, y=y, sigma=sigma).eval(),
                          [[6.065307e+01, 6.065307e+01],
                           [2.576757e-16, 3.726653e-04]])
      x = tf.constant([[10.0, 3.0], [10.0, 3.0]])
      y = tf.constant([[2.0, 3.0], [2.0, 3.0]])
      sigma = tf.constant(10.0)
      l = tf.constant(5.0)
      self.assertAllClose(rbf(x, y=y, sigma=sigma, l=l).eval(),
                          [[27.80373, 100], [27.80373, 100]])

  def test_contraint_raises(self):
    with self.test_session():
      x = tf.constant(0.0)
      y = tf.constant(1.0)
      sigma = tf.constant(-1.0)
      l = tf.constant(-5.0)
      with self.assertRaisesOpError('Condition'):
        rbf(x, y=y, sigma=sigma).eval()
        rbf(x, y=y, l=l).eval()
        rbf(x, y=y, sigma=sigma, l=l).eval()
      x = np.inf * tf.constant(1.0)
      y = tf.constant(1.0)
      sigma = tf.constant(1.0)
      l = tf.constant(5.0)
      with self.assertRaisesOpError('Inf'):
        rbf(x).eval()
        rbf(x, y=y).eval()
        rbf(x, y=y, sigma=sigma).eval()
        rbf(x, y=y, l=l).eval()
        rbf(x, y=y, sigma=sigma, l=l).eval()
      x = tf.constant(0.0)
      y = np.nan * tf.constant(1.0)
      sigma = tf.constant(1.0)
      l = tf.constant(5.0)
      with self.assertRaisesOpError('NaN'):
        rbf(x, y=y).eval()
        rbf(x, y=y, sigma=sigma).eval()
        rbf(x, y=y, l=l).eval()
        rbf(x, y=y, sigma=sigma, l=l).eval()

if __name__ == '__main__':
  tf.test.main()
