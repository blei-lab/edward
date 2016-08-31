from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import log_sum_exp


class test_log_sum_exp_class(tf.test.TestCase):

  def test_log_sum_exp_1d(self):
    with self.test_session():
      x = tf.constant([-1.0, -2.0, -3.0, -4.0])
      self.assertAllClose(log_sum_exp(x).eval(),
                          -0.5598103014388045)

  def test_log_sum_exp_2d(self):
    with self.test_session():
      x = tf.constant([[-1.0], [-2.0], [-3.0], [-4.0]])
      self.assertAllClose(log_sum_exp(x).eval(),
                          -0.5598103014388045)
      x = tf.constant([[-1.0, -2.0], [-3.0, -4.0]])
      self.assertAllClose(log_sum_exp(x).eval(),
                          -0.5598103014388045)
      self.assertAllClose(log_sum_exp(x, 0).eval(),
                          np.array([-0.87307198895702742,
                                    -1.8730719889570275]))
      self.assertAllClose(log_sum_exp(x, 1).eval(),
                          np.array([-0.68673831248177708,
                                    -2.6867383124817774]))

  def test_all_finite_raises(self):
    with self.test_session():
      x = np.inf * tf.constant([-1.0, -2.0, -3.0, -4.0])
      with self.assertRaisesOpError('Inf'):
        log_sum_exp(x).eval()
      x = tf.constant([-1.0, np.nan, -3.0, -4.0])
      with self.assertRaisesOpError('NaN'):
        log_sum_exp(x).eval()

if __name__ == '__main__':
  tf.test.main()
