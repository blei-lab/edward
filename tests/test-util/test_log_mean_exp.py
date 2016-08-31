from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import log_mean_exp


class test_log_mean_exp_class(tf.test.TestCase):

  def test_log_mean_exp_1d(self):
    with self.test_session():
      x = tf.constant([-1.0, -2.0, -3.0, -4.0])
      self.assertAllClose(log_mean_exp(x).eval(),
                          -1.9461046625586951)

  def test_log_mean_exp_2d(self):
    with self.test_session():
      x = tf.constant([[-1.0], [-2.0], [-3.0], [-4.0]])
      self.assertAllClose(log_mean_exp(x).eval(),
                          -1.9461046625586951)
      x = tf.constant([[-1.0, -2.0], [-3.0, -4.0]])
      self.assertAllClose(log_mean_exp(x).eval(),
                          -1.9461046625586951)
      self.assertAllClose(log_mean_exp(x, 0).eval(),
                          np.array([-1.5662191695169727,
                                    -2.5662191695169727]))
      self.assertAllClose(log_mean_exp(x, 1).eval(),
                          np.array([-1.3798854930417224,
                                    -3.3798854930417224]))

  def test_all_finite_raises(self):
    with self.test_session():
      x = np.inf * tf.constant([-1.0, -2.0, -3.0, -4.0])
      with self.assertRaisesOpError('Inf'):
        log_mean_exp(x).eval()
      x = tf.constant([-1.0, np.nan, -3.0, -4.0])
      with self.assertRaisesOpError('NaN'):
        log_mean_exp(x).eval()

if __name__ == '__main__':
  tf.test.main()
