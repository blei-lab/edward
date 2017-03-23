from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.util.tensorflow import get_control_variate_coef


class test_get_control_variate_coef(tf.test.TestCase):

  def test_calculate_correct_coefficient(self):
    with self.test_session():
      f = tf.constant([1.0, 2.0, 3.0, 4.0])
      h = tf.constant([2.0, 3.0, 8.0, 1.0])
      self.assertAllClose(get_control_variate_coef(f, h).eval(),
                          0.03448276)

if __name__ == '__main__':
  tf.test.main()
