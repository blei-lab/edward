from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Normal
from edward.util import check_data


class test_check_data_class(tf.test.TestCase):

  def test(self):
    with self.test_session():
      x = Normal(0.0, 1.0)
      qx = Normal(0.0, 1.0)
      x_ph = tf.placeholder(tf.float32, [])

      check_data({x: tf.constant(0.0)})
      check_data({x: np.float64(0.0)})
      check_data({x: np.int64(0)})
      check_data({x: 0.0})
      check_data({x: 0})
      check_data({x: False})
      check_data({x: '0'})
      check_data({x: x_ph})
      check_data({x: qx})
      check_data({2.0 * x: tf.constant(0.0)})
      self.assertRaises(TypeError, check_data, {0.0: x})
      self.assertRaises(TypeError, check_data, {x: tf.zeros(5)})
      self.assertRaises(TypeError, check_data, {x_ph: x})
      self.assertRaises(TypeError, check_data, {x_ph: x})
      self.assertRaises(TypeError, check_data,
                        {x: tf.constant(0, tf.float64)})
      self.assertRaises(TypeError, check_data,
                        {x_ph: tf.constant(0.0)})

      x_vec = Normal(tf.constant([0.0]), tf.constant([1.0]))
      qx_vec = Normal(tf.constant([0.0]), tf.constant([1.0]))

      check_data({x_vec: qx_vec})
      check_data({x_vec: [0.0]})
      check_data({x_vec: [0]})
      check_data({x_vec: ['0']})
      self.assertRaises(TypeError, check_data, {x: qx_vec})

if __name__ == '__main__':
  tf.test.main()
