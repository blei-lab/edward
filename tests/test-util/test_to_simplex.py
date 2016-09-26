from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import to_simplex


class test_to_simplex_class(tf.test.TestCase):

  def test_to_simplex_1d(self):
    with self.test_session():
      x = tf.constant([0.0])
      self.assertAllClose(to_simplex(x).eval(),
                          [0.5, 0.5])
      x = tf.constant([0.0, 10.0])
      self.assertAllClose(to_simplex(x).eval(),
                          [3.333333e-01, 6.666363e-01, 3.027916e-05])

  def test_to_simplex_2d(self):
    with self.test_session():
      x = tf.constant([[0.0], [0.0]])
      self.assertAllClose(to_simplex(x).eval(),
                          [[0.5, 0.5], [0.5, 0.5]])
      x = tf.constant([[0.0, 10.0], [0.0, 10.0]])
      self.assertAllClose(to_simplex(x).eval(),
                          [[3.333333e-01, 6.666363e-01, 3.027916e-05],
                           [3.333333e-01, 6.666363e-01, 3.027916e-05]])

  def test_all_finite_raises(self):
    with self.test_session():
      x = tf.constant([12.5, np.inf])
      with self.assertRaisesOpError('Inf'):
        to_simplex(x).eval()
      x = tf.constant([12.5, np.nan])
      with self.assertRaisesOpError('NaN'):
        to_simplex(x).eval()

if __name__ == '__main__':
  tf.test.main()
