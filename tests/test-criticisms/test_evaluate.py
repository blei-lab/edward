from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


class test_evaluate_class(tf.test.TestCase):

  def test_metrics(self):
    tf.InteractiveSession()
    x = Normal(mu=0.0, sigma=1.0)
    x_data = tf.constant(0.0)
    ed.evaluate('mean_squared_error', {x: x_data}, n_samples=1)
    ed.evaluate(['mean_squared_error'], {x: x_data}, n_samples=1)
    ed.evaluate(['mean_squared_error', 'mean_absolute_error'],
                {x: x_data}, n_samples=1)
    self.assertRaises(NotImplementedError, ed.evaluate, 'hello world',
                      {x: x_data}, n_samples=1)

  def test_data(self):
    tf.InteractiveSession()
    x_ph = tf.placeholder(tf.float32, [])
    x = Normal(mu=x_ph, sigma=1.0)
    y = 2.0 * Normal(mu=0.0, sigma=1.0)
    x_data = tf.constant(0.0)
    x_ph_data = np.array(0.0)
    y_data = tf.constant(20.0)
    ed.evaluate('mean_squared_error', {x: x_data, x_ph: x_ph_data},
                n_samples=1)
    ed.evaluate('mean_squared_error', {y: y_data}, n_samples=1)

  def test_n_samples(self):
    tf.InteractiveSession()
    x = Normal(mu=0.0, sigma=1.0)
    x_data = tf.constant(0.0)
    ed.evaluate('mean_squared_error', {x: x_data}, n_samples=1)
    ed.evaluate('mean_squared_error', {x: x_data}, n_samples=50)

  def test_output_key(self):
    tf.InteractiveSession()
    x_ph = tf.placeholder(tf.float32, [])
    x = Normal(mu=x_ph, sigma=1.0)
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

if __name__ == '__main__':
  tf.test.main()
