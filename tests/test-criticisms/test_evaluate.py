from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


class test_evaluate_class(tf.test.TestCase):

  def test_metrics(self):
    with self.test_session():
      x = Normal(loc=0.0, scale=1.0)
      x_data = tf.constant(0.0)
      ed.evaluate('mean_squared_error', {x: x_data}, n_samples=1)
      ed.evaluate(['mean_squared_error'], {x: x_data}, n_samples=1)
      ed.evaluate(['mean_squared_error', 'mean_absolute_error'],
                  {x: x_data}, n_samples=1)
      self.assertRaises(TypeError, ed.evaluate, x, {x: x_data}, n_samples=1)
      self.assertRaises(NotImplementedError, ed.evaluate, 'hello world',
                        {x: x_data}, n_samples=1)

  def test_data(self):
    with self.test_session():
      x_ph = tf.placeholder(tf.float32, [])
      x = Normal(loc=x_ph, scale=1.0)
      y = 2.0 * Normal(loc=0.0, scale=1.0)
      x_data = tf.constant(0.0)
      x_ph_data = np.array(0.0)
      y_data = tf.constant(20.0)
      ed.evaluate('mean_squared_error', {x: x_data, x_ph: x_ph_data},
                  n_samples=1)
      ed.evaluate('mean_squared_error', {y: y_data}, n_samples=1)
      self.assertRaises(TypeError, ed.evaluate, 'mean_squared_error',
                        {'y': y_data}, n_samples=1)

  def test_n_samples(self):
    with self.test_session():
      x = Normal(loc=0.0, scale=1.0)
      x_data = tf.constant(0.0)
      ed.evaluate('mean_squared_error', {x: x_data}, n_samples=1)
      ed.evaluate('mean_squared_error', {x: x_data}, n_samples=5)
      self.assertRaises(TypeError, ed.evaluate, 'mean_squared_error',
                        {x: x_data}, n_samples='1')

  def test_output_key(self):
    with self.test_session():
      x_ph = tf.placeholder(tf.float32, [])
      x = Normal(loc=x_ph, scale=1.0)
      y = 2.0 * x
      x_data = tf.constant(0.0)
      x_ph_data = np.array(0.0)
      y_data = tf.constant(20.0)
      ed.evaluate('mean_squared_error', {x: x_data, x_ph: x_ph_data},
                  n_samples=1)
      ed.evaluate('mean_squared_error', {y: y_data, x_ph: x_ph_data},
                  n_samples=1)
      ed.evaluate('mean_squared_error', {x: x_data, y: y_data, x_ph: x_ph_data},
                  n_samples=1, output_key=x)
      self.assertRaises(KeyError, ed.evaluate, 'mean_squared_error',
                        {x: x_data, y: y_data, x_ph: x_ph_data}, n_samples=1)
      self.assertRaises(TypeError, ed.evaluate, 'mean_squared_error',
                        {x: x_data, y: y_data, x_ph: x_ph_data}, n_samples=1,
                        output_key='x')

if __name__ == '__main__':
  tf.test.main()
