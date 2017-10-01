from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal


class test_random_variable_gradients_class(tf.test.TestCase):

  def test_first_order(self):
    with self.test_session() as sess:
      x = Bernoulli(0.5)
      y = 2 * x
      z = tf.gradients(y, x)[0]
      self.assertEqual(z.eval(), 2)

  def test_second_order(self):
    with self.test_session() as sess:
      x = Normal(0.0, 1.0)
      y = 2 * (x ** 2)
      z = tf.gradients(y, x)[0]
      z = tf.gradients(z, x)[0]
      self.assertEqual(z.eval(), 4.0)

if __name__ == '__main__':
  tf.test.main()
